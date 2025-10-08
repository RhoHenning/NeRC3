import torch
import numpy as np
import utils

class Bitset:
    def __init__(self, bit_size, device):
        self.bit_size = bit_size
        self.int8_size = bit_size // 8 + 1
        self.bits = torch.zeros((self.int8_size,), dtype=torch.int8, device=device)

    def add(self, indices):
        assert (indices < self.bit_size).all()
        mask = indices // 8
        bit_pos = (indices - mask * 8).to(torch.int8)
        self.bits.index_add_(0, mask, torch.ones_like(bit_pos) << bit_pos)

    def test(self, indices):
        assert (indices < self.bit_size).all()
        mask = indices // 8
        bit_pos = indices - mask * 8
        results = (self.bits[mask] >> bit_pos) & 1
        return results

class GeometrySampler:
    def __init__(self, **kwargs):
        depth = kwargs['depth']
        block_depth = kwargs['block_depth']
        batch_size = kwargs['batch_size']
        device = kwargs['device']
        self.device = device
        self.depth = depth
        self.width = 1 << depth
        self.block_depth = block_depth
        self.block_width = 1 << (depth - block_depth)
        self.batch_size = batch_size

    def local_voxels(self):
        base = torch.arange(self.block_width, dtype=torch.float32, device=self.device).unsqueeze(dim=1)
        i = base.repeat_interleave(self.block_width ** 2, dim=0)
        j = base.repeat_interleave(self.block_width, dim=0).repeat(self.block_width, 1)
        k = base.repeat(self.block_width ** 2, 1)
        voxels = torch.cat([i, j, k], dim=-1)
        return voxels
    
    def normalize(self, voxels):
        pass

class GeometryEncodingSampler(GeometrySampler):
    def __init__(self, **kwargs):
        occupied_ratio = kwargs['occupied_ratio']
        super().__init__(**kwargs)
        self.points = None
        self.blocks = None
        self.num_points = None
        self.num_blocks = None
        self.num_voxels = None
        self.occupied_ratio = occupied_ratio

    def init_batch_size(self):
        real_occupied_ratio = self.num_points / self.num_voxels
        if self.occupied_ratio:
            point_ratio = (self.occupied_ratio - real_occupied_ratio) / (1 - real_occupied_ratio)
            point_batch_size = int(self.batch_size * point_ratio)
            voxel_batch_size = self.batch_size - point_batch_size
            assert point_batch_size > 0 and voxel_batch_size > 0
            self.point_batch_size = point_batch_size
            self.voxel_batch_size = voxel_batch_size
            self.alpha = 1 - self.occupied_ratio
        else:
            self.alpha = 1 - real_occupied_ratio

    def init_bitset(self):
        self.bitset = Bitset(self.num_voxels, self.device)
        num_iters = int(np.ceil(self.num_points / self.batch_size))
        for iter in range(num_iters):
            start = iter * self.batch_size
            end = min(start + self.batch_size, self.num_points)
            flattend_points = self.flatten_voxels(self.points[start:end])
            self.bitset.add(flattend_points)

    def flatten_1(self, local_voxels):
        return (local_voxels[:, 0] * self.block_width + local_voxels[:, 1]) * self.block_width + local_voxels[:, 2]

    def flatten_2(self, local_voxel_indices, block_indices):
        return block_indices * (self.block_width ** 3) + local_voxel_indices

    def flatten_voxels(self, voxels):
        voxel_blocks = voxels.clone().to(torch.int64)
        voxel_blocks[:, -3:] //= self.block_width
        dist = torch.sum((self.blocks.repeat(len(voxel_blocks), 1, 1) - voxel_blocks.unsqueeze(dim=1)) ** 2, dim=-1)
        pair = torch.nonzero(dist == 0)
        block_indices = torch.zeros((len(voxels),), dtype=torch.int64, device=self.device)
        block_indices[pair[:, 0]] = pair[:, 1]
        local_voxels = voxels[:, -3:].to(torch.int64) % self.block_width
        local_voxel_indices = self.flatten_1(local_voxels)
        flattened_voxels = self.flatten_2(local_voxel_indices, block_indices)
        return flattened_voxels
    
    def sample_points(self, batch_size):
        point_indices = torch.randint(0, self.num_points, (batch_size,), device=self.device)
        points = self.points[point_indices]
        flattened_points = self.flatten_voxels(points)
        return points, flattened_points
    
    def sample_voxels(self, batch_size):
        block_indices = torch.randint(0, self.num_blocks, (batch_size,), device=self.device)
        blocks = self.blocks[block_indices]
        local_voxels = torch.randint(0, self.block_width, (batch_size, 3), device=self.device)
        local_voxel_indices = self.flatten_1(local_voxels)
        voxels = local_voxels + self.block_width * blocks[:, -3:]
        if blocks.shape[-1] == 4:
            frame_indices = blocks[:, 0].unsqueeze(dim=-1)
            voxels = torch.cat([frame_indices, voxels], dim=-1)
        flattened_voxels = self.flatten_2(local_voxel_indices, block_indices)
        return voxels, flattened_voxels
    
    def sample(self, normalize=True):
        if self.occupied_ratio:
            points, flattened_points = self.sample_points(self.point_batch_size)
            voxels, flattened_voxels = self.sample_voxels(self.voxel_batch_size)
            samples = torch.cat([points, voxels], dim=0)
            flattened_samples = torch.cat([flattened_points, flattened_voxels], dim=0)
        else:
            samples, flattened_samples = self.sample_voxels(self.batch_size)
        occupancies = self.bitset.test(flattened_samples)
        if normalize:
            samples = self.normalize(samples)
        return samples, occupancies

    def loss_function(self, probs, occupancies, gamma=2.0, eps=1e-5):
        probs = probs.squeeze(dim=-1)
        alpha_t = self.alpha * occupancies + (1 - self.alpha) * (1 - occupancies)
        probs_t = probs * occupancies + (1 - probs) * (1 - occupancies)
        loss = -alpha_t * ((1 - probs_t) ** gamma) * torch.log(probs_t + eps)
        return loss.mean()

class GeometryEncodingSampler3D(GeometryEncodingSampler):
    def __init__(self, points, **kwargs):
        depth = kwargs['depth']
        block_depth = kwargs['block_depth']
        device = kwargs['device']
        super().__init__(**kwargs)
        blocks = utils.partition_blocks(points, depth, block_depth)
        self.points = torch.from_numpy(points).to(device)
        self.blocks = torch.from_numpy(blocks).to(device)
        self.num_points = len(self.points)
        self.num_blocks = len(self.blocks)
        self.num_voxels = len(self.blocks) * (self.block_width ** 3)
        self.init_batch_size()
        self.init_bitset()
    
    def normalize(self, voxels):
        voxels = utils.normalize(voxels, min_bound=0, max_bound=self.width - 1)
        return voxels

class GeometryEncodingSampler4D(GeometryEncodingSampler):
    def __init__(self, points_list, **kwargs):
        depth = kwargs['depth']
        block_depth = kwargs['block_depth']
        device = kwargs['device']
        super().__init__(**kwargs)
        self.num_frames = len(points_list)
        self.points = []
        self.blocks = []
        self.cumsum_points = np.zeros((self.num_frames + 1,), dtype=np.int64)
        self.cumsum_blocks = np.zeros((self.num_frames + 1,), dtype=np.int64)
        for frame_index, points in enumerate(points_list):
            blocks = utils.partition_blocks(points, depth, block_depth)
            point_frames = np.full((len(points), 1), frame_index, dtype=np.float32)
            self.points.append(np.concatenate([point_frames, points], axis=-1))
            block_frames = np.full((len(blocks), 1), frame_index, dtype=np.float32)
            self.blocks.append(np.concatenate([block_frames, blocks], axis=-1))
            self.cumsum_points[frame_index + 1] = self.cumsum_points[frame_index] + len(points)
            self.cumsum_blocks[frame_index + 1] = self.cumsum_blocks[frame_index] + len(blocks)
        self.points = np.concatenate(self.points, axis=0)
        self.blocks = np.concatenate(self.blocks, axis=0)
        self.points = torch.from_numpy(self.points).to(device)
        self.blocks = torch.from_numpy(self.blocks).to(device)
        self.cumsum_points = torch.from_numpy(self.cumsum_points).to(device)
        self.cumsum_blocks = torch.from_numpy(self.cumsum_blocks).to(device)
        self.num_points = len(self.points)
        self.num_blocks = len(self.blocks)
        self.num_voxels = len(self.blocks) * (self.block_width ** 3)
        self.init_batch_size()
        self.init_bitset()

    def points_in_frame(self, frame_index, return_3d=True):
        start = self.cumsum_points[frame_index]
        end = self.cumsum_points[frame_index + 1]
        points = self.points[start:end]
        if return_3d:
            points = points[:, -3:]
        return points

    def blocks_in_frame(self, frame_index, return_3d=True):
        start = self.cumsum_blocks[frame_index]
        end = self.cumsum_blocks[frame_index + 1]
        blocks = self.blocks[start:end]
        if return_3d:
            blocks = blocks[:, -3:]
        return blocks
    
    def normalize(self, voxels):
        min_bound = torch.tensor([0, 0, 0, 0], device=self.device)
        max_bound = torch.tensor([self.num_frames - 1] + [self.width - 1] * 3, device=self.device)
        voxels = utils.normalize(voxels, min_bound, max_bound)
        return voxels

class GeometryDecodingSampler(GeometrySampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = None
        self.num_blocks = None
        self.num_voxels = None

class GeometryDecodingSampler3D(GeometryDecodingSampler):
    def __init__(self, blocks, **kwargs):
        device = kwargs['device']
        super().__init__(**kwargs)
        self.blocks = torch.from_numpy(blocks).to(device)
        self.num_blocks = len(self.blocks)
        self.num_voxels = len(self.blocks) * (self.block_width ** 3)
    
    def normalize(self, voxels):
        voxels = utils.normalize(voxels, min_bound=0, max_bound=self.width - 1)
        return voxels

class GeometryDecodingSampler4D(GeometryDecodingSampler):
    def __init__(self, num_blocks, blocks, **kwargs):
        device = kwargs['device']
        super().__init__(**kwargs)
        self.num_frames = len(num_blocks)
        self.cumsum_blocks = np.concatenate([np.array([0]), num_blocks])
        self.cumsum_blocks = np.cumsum(self.cumsum_blocks)
        frame_indices = np.arange(self.num_frames, dtype=np.float32)
        frame_indices = np.repeat(frame_indices, num_blocks)[:, np.newaxis]
        self.blocks = np.concatenate([frame_indices, blocks], axis=-1)
        self.blocks = torch.from_numpy(self.blocks).to(device)
        self.cumsum_blocks = torch.from_numpy(self.cumsum_blocks).to(device)
        self.num_blocks = len(self.blocks)
        self.num_voxels = len(self.blocks) * (self.block_width ** 3)

    def blocks_in_frame(self, frame_index, return_3d=True):
        start = self.cumsum_blocks[frame_index]
        end = self.cumsum_blocks[frame_index + 1]
        blocks = self.blocks[start:end]
        if return_3d:
            blocks = blocks[:, -3:]
        return blocks
    
    def normalize(self, voxels):
        min_bound = torch.tensor([0, 0, 0, 0], device=self.device)
        max_bound = torch.tensor([self.num_frames - 1] + [self.width - 1] * 3, device=self.device)
        voxels = utils.normalize(voxels, min_bound, max_bound)
        return voxels

class AttributeSampler:
    def __init__(self, **kwargs):
        depth = kwargs['depth']
        batch_size = kwargs['batch_size']
        device = kwargs['device']
        self.device = device
        self.depth = depth
        self.width = 1 << depth
        self.batch_size = batch_size
    
    def normalize(self, voxels):
        pass

class AttributeEncodingSampler(AttributeSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voxels = None
        self.target_colors = None
        self.num_points = None
        self.num_voxels = None
    
    def sample(self, normalize=True):
        indices = torch.randint(0, self.num_voxels, (self.batch_size,), device=self.device)
        samples = self.voxels[indices]
        target_colors = self.target_colors[indices]
        if normalize:
            samples = self.normalize(samples)
        return samples, target_colors

    def loss_function(self, colors, reconstructed_colors):
        loss = torch.sum((reconstructed_colors - colors) ** 2, dim=-1)
        return loss.mean()

class AttributeEncodingSampler3D(AttributeEncodingSampler):
    def __init__(self, points, colors, reconstructed_points, **kwargs):
        knn = kwargs['knn']
        device = kwargs['device']
        super().__init__(**kwargs)
        self.num_points = len(points)
        voxels = reconstructed_points
        indices = utils.nearest_neighbor_indices(points, voxels, knn)
        target_colors = np.mean(colors[indices], axis=1)
        self.voxels = torch.from_numpy(voxels).to(device)
        self.target_colors = torch.from_numpy(target_colors).to(device)
        self.num_voxels = len(self.voxels)
    
    def normalize(self, voxels):
        voxels = utils.normalize(voxels, min_bound=0, max_bound=self.width - 1)
        return voxels

class AttributeEncodingSampler4D(AttributeEncodingSampler):
    def __init__(self, points_list, colors_list, reconstructed_points_list, **kwargs):
        knn = kwargs['knn']
        device = kwargs['device']
        super().__init__(**kwargs)
        self.num_frames = len(points_list)
        self.voxels = []
        self.target_colors = []
        self.num_points = 0
        self.cumsum_voxels = np.zeros((self.num_frames + 1,), dtype=np.int64)
        for frame_index in range(self.num_frames):
            points = points_list[frame_index]
            colors = colors_list[frame_index]
            voxels = reconstructed_points_list[frame_index]
            frame_indices = np.full((len(voxels), 1), frame_index, dtype=np.float32)
            self.voxels.append(np.concatenate([frame_indices, voxels], axis=-1))
            indices = utils.nearest_neighbor_indices(points, voxels, knn)
            target_colors = np.mean(colors[indices], axis=1)
            self.target_colors.append(target_colors)
            self.num_points += len(points)
            self.cumsum_voxels[frame_index + 1] = self.cumsum_voxels[frame_index] + len(voxels)
        self.voxels = np.concatenate(self.voxels, axis=0)
        self.target_colors = np.concatenate(self.target_colors, axis=0)
        self.voxels = torch.from_numpy(self.voxels).to(device)
        self.target_colors = torch.from_numpy(self.target_colors).to(device)
        self.cumsum_voxels = torch.from_numpy(self.cumsum_voxels).to(device)
        self.num_voxels = len(self.voxels)

    def voxels_in_frame(self, frame_index, return_3d=True):
        start = self.cumsum_voxels[frame_index]
        end = self.cumsum_voxels[frame_index + 1]
        voxels = self.voxels[start:end]
        if return_3d:
            voxels = voxels[:, -3:]
        return voxels
    
    def normalize(self, voxels):
        min_bound = torch.tensor([0, 0, 0, 0], device=self.device)
        max_bound = torch.tensor([self.num_frames - 1] + [self.width - 1] * 3, device=self.device)
        voxels = utils.normalize(voxels, min_bound, max_bound)
        return voxels

class AttributeDecodingSampler(AttributeSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voxels = None
        self.num_voxels = None
    
class AttributeDecodingSampler3D(AttributeDecodingSampler):
    def __init__(self, reconstructed_points, **kwargs):
        device = kwargs['device']
        super().__init__(**kwargs)
        self.voxels = torch.from_numpy(reconstructed_points).to(device)
        self.num_voxels = len(self.voxels)
    
    def normalize(self, voxels):
        voxels = utils.normalize(voxels, min_bound=0, max_bound=self.width - 1)
        return voxels
    
class AttributeDecodingSampler4D(AttributeDecodingSampler):
    def __init__(self, reconstructed_points_list, **kwargs):
        device = kwargs['device']
        super().__init__(**kwargs)
        self.num_frames = len(reconstructed_points_list)
        self.voxels = []
        self.cumsum_voxels = np.zeros((self.num_frames + 1,), dtype=np.int64)
        for frame_index in range(self.num_frames):
            voxels = reconstructed_points_list[frame_index]
            frame_indices = np.full((len(voxels), 1), frame_index, dtype=np.float32)
            self.voxels.append(np.concatenate([frame_indices, voxels], axis=-1))
            self.cumsum_voxels[frame_index + 1] = self.cumsum_voxels[frame_index] + len(voxels)
        self.voxels = np.concatenate(self.voxels, axis=0)
        self.voxels = torch.from_numpy(self.voxels).to(device)
        self.cumsum_voxels = torch.from_numpy(self.cumsum_voxels).to(device)
        self.num_voxels = len(self.voxels)

    def voxels_in_frame(self, frame_index, return_3d=True):
        start = self.cumsum_voxels[frame_index]
        end = self.cumsum_voxels[frame_index + 1]
        voxels = self.voxels[start:end]
        if return_3d:
            voxels = voxels[:, -3:]
        return voxels
    
    def normalize(self, voxels):
        min_bound = torch.tensor([0, 0, 0, 0], device=self.device)
        max_bound = torch.tensor([self.num_frames - 1] + [self.width - 1] * 3, device=self.device)
        voxels = utils.normalize(voxels, min_bound, max_bound)
        return voxels

def get_sampler(component, train, configs, device, **kwargs):
    points_list = kwargs.get('points_list', None)
    num_blocks = kwargs.get('num_blocks', None)
    blocks = kwargs.get('blocks', None)
    colors_list = kwargs.get('colors_list', None)
    reconstructed_points_list = kwargs.get('reconstructed_points_list', None)
    group_size = kwargs.get('group_size', configs['group_size'])

    assert component in ['geometry', 'attribute']
    kwargs = {
        'depth': configs['depth'],
        'block_depth': configs['block_depth'],
        'knn': configs['knn'],
        'batch_size': configs[component]['batch_size'],
        'device': device,
        'occupied_ratio': configs['geometry']['occupied_ratio']
    }
    if group_size == 1:
        if component == 'geometry':
            if train:
                sampler = GeometryEncodingSampler3D(points_list[0], **kwargs)
            else:
                sampler = GeometryDecodingSampler3D(blocks, **kwargs)
        elif component == 'attribute':
            if train:
                sampler = AttributeEncodingSampler3D(points_list[0], colors_list[0], reconstructed_points_list[0], **kwargs)
            else:
                sampler = AttributeDecodingSampler3D(reconstructed_points_list[0], **kwargs)
    else:
        if component == 'geometry':
            if train:
                sampler = GeometryEncodingSampler4D(points_list, **kwargs)
            else:
                sampler = GeometryDecodingSampler4D(num_blocks, blocks, **kwargs)
        elif component == 'attribute':
            if train:
                sampler = AttributeEncodingSampler4D(points_list, colors_list, reconstructed_points_list, **kwargs)
            else:
                sampler = AttributeDecodingSampler4D(reconstructed_points_list, **kwargs)
    return sampler