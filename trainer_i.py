import torch
import torch.optim as optim
import numpy as np

class TrainerCurve:
    def __init__(self, model, samplers, logger, fix_start, fix_end):
        self.model = model
        self.samplers = samplers
        self.num_frames = len(self.samplers)
        self.num_points = sum([sampler.num_points for sampler in self.samplers])
        self.logger = logger
        self.fix_start = fix_start
        self.fix_end = fix_end
    
    def train(self, **kwargs):
        lr = kwargs['lr']
        weight_decay = kwargs['weight_decay']
        num_steps = kwargs['num_steps']
        scheduler_steps = kwargs['scheduler_steps']
        gamma = kwargs['gamma']
        lmbda = kwargs['lmbda']
        check_interval = kwargs['check_interval']

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        step_size = num_steps // scheduler_steps
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        distortion_list = []
        loss_list = []
        self.model.train()
        for step in range(num_steps):
            frame_index = np.random.randint(int(self.fix_start), self.num_frames - int(self.fix_end))
            sampler = self.samplers[frame_index]
            t = frame_index / (self.num_frames - 1)

            samples, targets = sampler.sample()
            outputs = self.model(samples, t)
            distortion = sampler.loss_function(outputs, targets)
            penalty = self.model.l1_penalty() / self.num_points
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
                                f'sparsity: {self.model.sparsity():.4}')
                distortion_list.clear()
                loss_list.clear()
            scheduler.step()