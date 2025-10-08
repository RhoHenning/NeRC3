import torch
import torch.optim as optim
import numpy as np
import utils

class Trainer:
    def __init__(self, model, sampler, logger):
        self.model = model
        self.sampler = sampler
        self.logger = logger
    
    def train(self, **kwargs):
        lr = kwargs['lr']
        weight_decay = kwargs['weight_decay']
        num_steps = kwargs['num_steps']
        scheduler_steps = kwargs['scheduler_steps']
        gamma = kwargs['gamma']
        lmbda = kwargs['lmbda']
        check_interval = kwargs['check_interval']
        reference_model = kwargs['reference_model']

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        step_size = num_steps // scheduler_steps
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        distortion_list = []
        loss_list = []
        self.model.train()
        for step in range(num_steps):
            samples, targets = self.sampler.sample()
            outputs = self.model(samples)
            distortion = self.sampler.loss_function(outputs, targets)
            penalty = self.model.l1_penalty(reference_model) / self.sampler.num_points
            loss = distortion + lmbda * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            distortion_list.append(distortion.item())
            loss_list.append(loss.item())
            if (step + 1) % check_interval == 0:
                mean_distortion = sum(distortion_list) / len(distortion_list)
                mean_loss = sum(loss_list) / len(loss_list)
                check = (step + 1) // check_interval
                num_checks = num_steps // check_interval
                self.logger.log(f'[Check {check}/{num_checks}] ' + 
                                f'lr: {scheduler.get_last_lr()[0]:.1} ' + 
                                f'loss: {mean_loss:.4} ' +
                                f'distortion: {mean_distortion:.4} ' +
                                f'sparsity: {self.model.sparsity(reference_model):.4}')
                distortion_list.clear()
                loss_list.clear()
            scheduler.step()
    
    @torch.no_grad()
    def evaluate_loss(self, num_steps):
        distortion_list = []
        self.model.train()
        for _ in range(num_steps):
            samples, targets = self.sampler.sample()
            outputs = self.model(samples)
            distortion = self.sampler.loss_function(outputs, targets)
            distortion_list.append(distortion.item())
        
        mean_distortion = sum(distortion_list) / len(distortion_list)
        sparsity = self.model.sparsity()
        return mean_distortion, sparsity

    @torch.no_grad()
    def reconstruct_geometry(self, frame_index, threshold):
        self.model.eval()
        reconstructed_points = []
        local_voxels = self.sampler.local_voxels()
        if self.sampler.blocks.shape[-1] == 3:
            blocks = self.sampler.blocks
        else:
            blocks = self.sampler.blocks_in_frame(frame_index)
            frame_indices = torch.full((len(local_voxels), 1), frame_index, dtype=torch.float32, device=self.sampler.device)
            local_voxels = torch.cat([frame_indices, local_voxels], dim=-1)
        for block in blocks:
            voxels = local_voxels.clone()
            voxels[:, -3:] += block * self.sampler.block_width
            normalized_voxels = self.sampler.normalize(voxels)
            num_iters = int(np.ceil(len(voxels) / self.sampler.batch_size))
            for iter in range(num_iters):
                start = iter * self.sampler.batch_size
                end = min(start + self.sampler.batch_size, len(voxels))
                samples = normalized_voxels[start:end]
                probs = self.model(samples)
                mask = probs.squeeze(dim=-1) > threshold
                reconstructed_points.append(voxels[start:end][mask][:, -3:])
        reconstructed_points = torch.cat(reconstructed_points, dim=0)
        reconstructed_points = reconstructed_points.cpu().numpy()
        return reconstructed_points
    
    @torch.no_grad()
    def reconstruct_attribute(self, frame_index):
        self.model.eval()
        reconstructed_colors = []
        if self.sampler.voxels.shape[-1] == 3:
            voxels = self.sampler.voxels
        else:
            voxels = self.sampler.voxels_in_frame(frame_index, return_3d=False)
        normalized_voxels = self.sampler.normalize(voxels)
        num_iters = int(np.ceil(len(voxels) / self.sampler.batch_size))
        for iter in range(num_iters):
            start = iter * self.sampler.batch_size
            end = min(start + self.sampler.batch_size, len(voxels))
            samples = normalized_voxels[start:end]
            colors = self.model(samples)
            reconstructed_colors.append(colors)
        reconstructed_colors = torch.cat(reconstructed_colors, dim=0)
        reconstructed_colors = reconstructed_colors.cpu().numpy()
        return reconstructed_colors
    
    def chamfer_distance(self, frame_index, threshold):
        if self.sampler.points.shape[-1] == 3:
            points = self.sampler.points.cpu().numpy()
        else:
            points = self.sampler.points_in_frame(frame_index).cpu().numpy()
        reconstructed_points = self.reconstruct_geometry(frame_index, threshold)
        if len(reconstructed_points) == 0:
            return np.inf
        dist = utils.chamfer_distance(points, reconstructed_points)
        return dist

    def finetune_threshold(self, frame_index, num_steps):
        left, right = 0.0, 1.0
        dist_1 = None
        dist_2 = None
        for step in range(num_steps):
            mid_1 = left + (3 - np.sqrt(5)) / 2 * (right - left)
            mid_2 = left + (np.sqrt(5) - 1) / 2 * (right - left)
            if dist_1 is None:
                dist_1 = self.chamfer_distance(frame_index, mid_1)
                self.logger.log(f'[Step {step + 1}/{num_steps}] ' + 
                                f'threshold: {mid_1:.6} ' + 
                                f'chamfer distance: {dist_1:.6}')
            if dist_2 is None:
                dist_2 = self.chamfer_distance(frame_index, mid_2)
                self.logger.log(f'[Step {step + 1}/{num_steps}] ' + 
                                f'threshold: {mid_2:.6} ' + 
                                f'chamfer distance: {dist_2:.6}')
            if dist_2 != np.inf and dist_1 > dist_2:
                left = mid_1
                dist_1 = dist_2
                dist_2 = None
            else:
                right = mid_2
                dist_2 = dist_1
                dist_1 = None
        threshold = (left + right) / 2
        return threshold

def get_train_kwargs(component, configs, reference_model=None):
    assert component in ['geometry', 'attribute']
    kwargs = {
        'lr': configs[component]['lr'],
        'weight_decay': configs[component]['weight_decay'],
        'num_steps': configs[component]['train_steps'],
        'scheduler_steps': configs[component]['scheduler_steps'],
        'gamma': configs[component]['gamma'],
        'lmbda': configs[component]['lmbda'],
        'check_interval': configs[component]['check_interval'],
        'reference_model': reference_model
    }
    return kwargs