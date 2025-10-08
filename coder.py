import os
import time
import torch
import numpy as np
import utils
from model import get_model
from sampler import get_sampler
from trainer import Trainer, get_train_kwargs

class Coder:
    def __init__(self, logger, configs, device):
        self.logger = logger
        self.configs = configs
        self.device = device

    def get_trainer(self, component, train, **kwargs):
        model = get_model(component, self.configs, self.device)
        sampler = get_sampler(component, train, self.configs, self.device, **kwargs)
        trainer = Trainer(model, sampler, self.logger)
        return trainer

    def split_bits(self, bits, num_frames):
        block_depth = self.configs['block_depth']
        threshold_bytes = utils.bits_to_bytes(bits[:32 * num_frames])
        thresholds = np.frombuffer(threshold_bytes, dtype=np.float32)
        bits = bits[32 * num_frames:]
        num_blocks = utils.bits_to_digits(bits[:block_depth * 3 * num_frames], depth=block_depth * 3)
        sum_blocks = np.sum(num_blocks)
        bits = bits[block_depth * 3 * num_frames:]
        blocks = utils.bits_to_digits(bits[:block_depth * 3 * sum_blocks], depth=block_depth).reshape(-1, 3).astype(np.float32)
        bits = bits[block_depth * 3 * sum_blocks:]
        bits = bits[len(bits) % 8:]
        model_bytes = utils.bits_to_bytes(bits)
        return thresholds, num_blocks, blocks, model_bytes
    
    def merge_bits(self, thresholds, num_blocks, blocks, model_bytes):
        block_depth = self.configs['block_depth']
        threshold_bytes = np.array(thresholds, dtype=np.float32).tobytes()
        threshold_bits = utils.bytes_to_bits(threshold_bytes)
        num_blocks = utils.digits_to_bits(num_blocks, depth=block_depth * 3)
        blocks = utils.digits_to_bits(blocks.reshape(-1).astype(np.int64), depth=block_depth)
        block_bits = np.concatenate([num_blocks, blocks])
        model_bits = utils.bytes_to_bits(model_bytes)
        if len(block_bits) % 8 != 0:
            block_bits = np.concatenate([block_bits, np.zeros(8 - len(block_bits) % 8, dtype=np.int64)])
        bits = np.concatenate([threshold_bits, block_bits, model_bits])
        return bits
    
    def encode_geometry(self, points_list, start_frame, encode_colors, reference_model=None):
        exp_dir = self.configs['exp_dir']
        finetune_steps = self.configs['geometry']['finetune_steps']
        encode_diff = self.configs['encode_diff']
        num_frames = len(points_list)
        if num_frames == 1:
            suffix = f'{start_frame:04d}'
        else:
            suffix = f'{start_frame:04d}-{start_frame + num_frames - 1:04d}'
        model_dir = os.path.join(exp_dir, 'models')
        model_path = os.path.join(model_dir, f'model_{suffix}.pt')
        threshold_path = os.path.join(model_dir, f'model_{suffix}_th.npy')

        trainer = self.get_trainer('geometry', train=True, points_list=points_list)
        if os.path.exists(model_path):
            self.logger.log('Load existing model...')
            trainer.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            self.logger.log('Train model from scratch...')
            trainer.train(**get_train_kwargs('geometry', self.configs, reference_model))
            torch.save(trainer.model.state_dict(), model_path)
        self.logger.log('Encode and decode model...')
        model_bytes = trainer.model.encode(reference_model)
        trainer.model.decode(model_bytes, reference_model)

        if os.path.exists(threshold_path):
            self.logger.log('Load existing threshold...')
            thresholds = np.load(threshold_path)
        else:
            thresholds = np.empty((num_frames,))
            for frame_index in range(num_frames):
                self.logger.log(f'Finetune thresholds ({frame_index + 1}/{num_frames})...')
                thresholds[frame_index] = trainer.finetune_threshold(frame_index, finetune_steps)
            np.save(threshold_path, thresholds)
        blocks = trainer.sampler.blocks[:, -3:].cpu().numpy()
        if trainer.sampler.blocks.shape[-1] == 4:
            cumsum_blocks = trainer.sampler.cumsum_blocks.cpu().numpy()
            num_blocks = cumsum_blocks[1:] - cumsum_blocks[:-1]
        else:
            num_blocks = np.array([len(blocks)])
        bits = self.merge_bits(thresholds, num_blocks, blocks, model_bytes)
        
        if encode_colors:
            reconstructed_points_list = []
            for frame_index, threshold in enumerate(thresholds):
                frame = start_frame + frame_index
                self.logger.log(f'Reconstruct frame {frame:04d} ({frame_index + 1}/{num_frames})...')
                reconstructed_points = trainer.reconstruct_geometry(frame_index, threshold)
                reconstructed_points_list.append(reconstructed_points)
        else:
            reconstructed_points_list = [None] * num_frames
        reference_model = trainer.model if encode_diff else None
        return bits, reconstructed_points_list, reference_model

    def encode_attribute(self, points_list, colors_list, reconstructed_points_list, start_frame, reference_model=None):
        exp_dir = self.configs['exp_dir']
        encode_diff = self.configs['encode_diff']
        num_frames = len(points_list)
        if num_frames == 1:
            suffix = f'{start_frame:04d}'
        else:
            suffix = f'{start_frame:04d}-{start_frame + num_frames - 1:04d}'

        model_dir = os.path.join(exp_dir, 'models')
        attr_model_path = os.path.join(model_dir, f'attr_model_{suffix}.pt')
        trainer = self.get_trainer('attribute', train=True, points_list=points_list, colors_list=colors_list, reconstructed_points_list=reconstructed_points_list)
        if os.path.exists(attr_model_path):
            self.logger.log('Load existing model...')
            trainer.model.load_state_dict(torch.load(attr_model_path, map_location=self.device))
        else:
            self.logger.log('Train model from scratch...')
            trainer.train(**get_train_kwargs('attribute', self.configs, reference_model))
            torch.save(trainer.model.state_dict(), attr_model_path)
        self.logger.log('Encode and decode model...')
        model_bytes = trainer.model.encode(reference_model)
        trainer.model.decode(model_bytes, reference_model)
        bits = utils.bytes_to_bits(model_bytes)
        reference_model = trainer.model if encode_diff else None
        return bits, reference_model

    def encode(self, encode_colors):
        dataset_dir = self.configs['dataset_dir']
        exp_dir = self.configs['exp_dir']
        sequence = self.configs['sequence']
        global_num_frames = self.configs['num_frames']
        global_start_frame = self.configs['start_frame']
        group_size = self.configs['group_size']
        bin_path = os.path.join(exp_dir, 'encoded.bin')

        points_list = []
        colors_list = []
        for frame_index in range(global_num_frames):
            frame = global_start_frame + frame_index
            self.logger.log(f'Load frame {frame:04d} ({frame_index + 1}/{global_num_frames})...')
            cloud_path = os.path.join(dataset_dir, f'{sequence}_{frame:04d}.ply')
            assert os.path.exists(cloud_path)
            points, colors, _ = utils.load_ply_cloud(cloud_path)
            points_list.append(points)
            colors_list.append(colors)
        
        encode_time = time.time()
        bits_list = []
        num_groups = int(np.ceil(global_num_frames / group_size))
        reference_model = None
        attr_reference_model = None
        for group in range(num_groups):
            start = group * group_size
            end = min(start + group_size, global_num_frames)
            start_frame = global_start_frame + start
            end_frame = global_start_frame + end
            if end_frame - start_frame == 1:
                suffix = f'{start_frame:04d}'
            else:
                suffix = f'{start_frame:04d}-{end_frame - 1:04d}'
            self.logger.log(f'Encode frame {suffix} ({group + 1}/{num_groups})...')

            self.logger.log('Encode geometry...')
            group_bits, reconstructed_points_list, reference_model = self.encode_geometry(points_list[start:end], start_frame, encode_colors, reference_model)
            num_group_bits = utils.digits_to_bits(np.array([len(group_bits)]), depth=32)
            bits_list.append(np.concatenate([num_group_bits, group_bits]))

            if encode_colors:
                self.logger.log('Encode attribute...')
                group_bits, attr_reference_model = self.encode_attribute(points_list[start:end], colors_list[start:end], reconstructed_points_list, start_frame, attr_reference_model)
                num_group_bits = utils.digits_to_bits(np.array([len(group_bits)]), depth=32)
                bits_list.append(np.concatenate([num_group_bits, group_bits]))
        
        bits = np.concatenate(bits_list)
        utils.write_binary(bits, bin_path)
        encode_time = time.time() - encode_time
        return encode_time
        
    def decode_geometry(self, bits, num_frames, start_frame, reference_model=None):
        encode_diff = self.configs['encode_diff']
        thresholds, num_blocks, blocks, model_bytes = self.split_bits(bits, num_frames)
        trainer = self.get_trainer('geometry', train=False, num_blocks=num_blocks, blocks=blocks)
        trainer.model.decode(model_bytes, reference_model)
        reconstructed_points_list = []
        for frame_index, threshold in enumerate(thresholds):
            frame = start_frame + frame_index
            self.logger.log(f'Reconstruct frame {frame:04d} ({frame_index + 1}/{num_frames})...')
            reconstructed_points = trainer.reconstruct_geometry(frame_index, threshold)
            reconstructed_points_list.append(reconstructed_points)
        reference_model = trainer.model if encode_diff else None
        return reconstructed_points_list, reference_model
    
    def decode_attribute(self, reconstructed_points_list, bits, start_frame, reference_model=None):
        encode_diff = self.configs['encode_diff']
        num_frames = len(reconstructed_points_list)
        model_bytes = utils.bits_to_bytes(bits)
        trainer = self.get_trainer('attribute', train=False, reconstructed_points_list=reconstructed_points_list)
        trainer.model.decode(model_bytes, reference_model)
        reconstructed_colors_list = []
        for frame_index in range(num_frames):
            frame = start_frame + frame_index
            self.logger.log(f'Reconstruct frame {frame:04d} ({frame_index + 1}/{num_frames})...')
            reconstructed_colors = trainer.reconstruct_attribute(frame_index)
            reconstructed_colors_list.append(reconstructed_colors)
        reference_model = trainer.model if encode_diff else None
        return reconstructed_colors_list, reference_model
    
    def decode(self, encode_colors):
        exp_dir = self.configs['exp_dir']
        global_num_frames = self.configs['num_frames']
        global_start_frame = self.configs['start_frame']
        group_size = self.configs['group_size']
        reconstructed_dir = os.path.join(exp_dir, 'reconstructed')
        bin_path = os.path.join(exp_dir, 'encoded.bin')
        assert os.path.exists(bin_path)

        decode_time = time.time()
        global_reconstructed_points_list = []
        global_reconstructed_colors_list = []
        bits = utils.load_binary(bin_path)
        num_groups = int(np.ceil(global_num_frames / group_size))
        reference_model = None
        attr_reference_model = None
        for group in range(num_groups):
            start = group * group_size
            end = min(start + group_size, global_num_frames)
            start_frame = global_start_frame + start
            end_frame = global_start_frame + end
            if end_frame - start_frame == 1:
                suffix = f'{start_frame:04d}'
            else:
                suffix = f'{start_frame:04d}-{end_frame - 1:04d}'
            self.logger.log(f'Decode frame {suffix} ({group + 1}/{num_groups})...')

            self.logger.log('Decode geometry...')
            num_group_bits = utils.bits_to_digits(bits[:32], depth=32).squeeze()
            group_bits = bits[32:32 + num_group_bits]
            reconstructed_points_list, reference_model = self.decode_geometry(group_bits, end - start, start_frame, reference_model)
            bits = bits[32 + num_group_bits:]

            if encode_colors:
                self.logger.log('Decode attribute...')
                num_group_bits = utils.bits_to_digits(bits[:32], depth=32).squeeze()
                group_bits = bits[32:32 + num_group_bits]
                reconstructed_colors_list, attr_reference_model = self.decode_attribute(reconstructed_points_list, group_bits, start_frame, attr_reference_model)
                bits = bits[32 + num_group_bits:]
            else:
                reconstructed_colors_list = [None] * global_num_frames
            global_reconstructed_points_list.extend(reconstructed_points_list)
            global_reconstructed_colors_list.extend(reconstructed_colors_list)
        decode_time = time.time() - decode_time
        
        for frame_index in range(global_num_frames):
            frame = global_start_frame + frame_index
            self.logger.log(f'Write frame {frame:04d} ({frame_index + 1}/{global_num_frames})...')
            reconstructed_path = os.path.join(reconstructed_dir, f'reconstructed_{frame:04d}.ply')
            reconstructed_points = global_reconstructed_points_list[frame_index]
            reconstructed_colors = global_reconstructed_colors_list[frame_index]
            utils.write_ply_cloud(reconstructed_path, reconstructed_points, reconstructed_colors)
        return decode_time