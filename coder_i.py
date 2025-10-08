import os
import time
import torch
import numpy as np
import utils
from model_i import get_model
from sampler import get_sampler
from trainer import Trainer, get_train_kwargs
from trainer_i import TrainerCurve

class CoderCurve:
    def __init__(self, logger, configs, device):
        self.logger = logger
        self.configs = configs
        self.device = device

    def get_trainer(self, component, train, **kwargs):
        fix_start = self.configs['fix_start']
        fix_end = self.configs['fix_end']
        points_list = kwargs.get('points_list', None)
        num_blocks = kwargs.get('num_blocks', None)
        blocks = kwargs.get('blocks', None)
        colors_list = kwargs.get('colors_list', None)
        reconstructed_points_list = kwargs.get('reconstructed_points_list', None)
        num_frames = kwargs['num_frames']
        model = kwargs.get('model', None)
        start_model = kwargs.get('start_model', None)
        end_model = kwargs.get('end_model', None)

        if num_frames == 1:
            if model is None:
                model = get_model(component, self.configs, self.device)
            sampler = get_sampler(component, train, self.configs, self.device, group_size=1, **kwargs)
            trainer = Trainer(model, sampler, self.logger)
        else:
            samplers = []
            start = 0
            for frame_index in range(num_frames):
                kwargs_t = {}
                if points_list is not None:
                    kwargs_t['points_list'] = [points_list[frame_index]]
                if colors_list is not None:
                    kwargs_t['colors_list'] = [colors_list[frame_index]]
                if reconstructed_points_list is not None:
                    kwargs_t['reconstructed_points_list'] = [reconstructed_points_list[frame_index]]
                if blocks is not None:
                    end = start + num_blocks[frame_index]
                    kwargs_t['blocks'] = blocks[start:end]
                    start = end
                sampler = get_sampler(component, train, self.configs, self.device, group_size=1, **kwargs_t)
                samplers.append(sampler)
            model = get_model(component, self.configs, self.device, curve=True, start_model=start_model, end_model=end_model)
            trainer = TrainerCurve(model, samplers, self.logger, fix_start, fix_end)
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
    
    def encode_geometry(self, points_list, start_frame, encode_colors, early_return=False):
        exp_dir = self.configs['exp_dir']
        fix_start = self.configs['fix_start']
        fix_end = self.configs['fix_end']
        finetune_steps = self.configs['geometry']['finetune_steps']
        num_frames = len(points_list)
        model_dir = os.path.join(exp_dir, 'models')
        start_model = None
        end_model = None
        if num_frames == 1:
            suffix = f'{start_frame:04d}'
        else:
            end_frame = start_frame + num_frames - 1
            if fix_start:
                start_model = get_model('geometry', self.configs, self.device)
                start_model_path = os.path.join(model_dir, f'model_{start_frame:04d}.pt')
                assert os.path.exists(start_model_path)
                start_model.load_state_dict(torch.load(start_model_path, map_location=self.device))
            if fix_end:
                end_model = get_model('geometry', self.configs, self.device)
                end_model_path = os.path.join(model_dir, f'model_{end_frame:04d}.pt')
                assert os.path.exists(end_model_path)
                end_model.load_state_dict(torch.load(end_model_path, map_location=self.device))
            suffix = f'{start_frame:04d}-{end_frame:04d}'
        model_path = os.path.join(model_dir, f'model_{suffix}.pt')
        threshold_path = os.path.join(model_dir, f'model_{suffix}_th.npy')

        trainer = self.get_trainer('geometry', train=True, num_frames=num_frames, points_list=points_list, start_model=start_model, end_model=end_model)
        if os.path.exists(model_path):
            self.logger.log('Load existing model...')
            trainer.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            self.logger.log('Train model from scratch...')
            trainer.train(**get_train_kwargs('geometry', self.configs))
            torch.save(trainer.model.state_dict(), model_path)
        if early_return:
            return
        self.logger.log('Encode and decode model...')
        model_bytes = trainer.model.encode()
        trainer.model.decode(model_bytes)

        if num_frames == 1:
            trainers = [trainer]
        else:
            trainers = []
            for frame_index in range(num_frames):
                model = get_model('geometry', self.configs, self.device)
                t = frame_index / (num_frames - 1)
                trainer.model.export_parameters_t(model, t)
                trainer_t = self.get_trainer('geometry', train=True, num_frames=1, points_list=[points_list[frame_index]], model=model)
                trainers.append(trainer_t)

        if os.path.exists(threshold_path):
            self.logger.log('Load existing threshold...')
            thresholds = np.load(threshold_path)
        else:
            thresholds = np.empty((num_frames,))
            for frame_index in range(num_frames):
                self.logger.log(f'Finetune thresholds ({frame_index + 1}/{num_frames})...')
                thresholds[frame_index] = trainers[frame_index].finetune_threshold(0, finetune_steps)
            np.save(threshold_path, thresholds)
        blocks_list = []
        num_blocks_list = []
        for frame_index in range(num_frames):
            blocks = trainers[frame_index].sampler.blocks.cpu().numpy()
            num_blocks = np.array([len(blocks)])
            blocks_list.append(blocks)
            num_blocks_list.append(num_blocks)
        blocks = np.concatenate(blocks_list, axis=0)
        num_blocks = np.concatenate(num_blocks_list)
        bits = self.merge_bits(thresholds, num_blocks, blocks, model_bytes)
        
        if encode_colors:
            reconstructed_points_list = []
            for frame_index in range(num_frames):
                frame = start_frame + frame_index
                self.logger.log(f'Reconstruct frame {frame:04d} ({frame_index + 1}/{num_frames})...')
                reconstructed_points = trainers[frame_index].reconstruct_geometry(frame_index, thresholds[frame_index])
                reconstructed_points_list.append(reconstructed_points)
        else:
            reconstructed_points_list = [None] * num_frames
        return bits, reconstructed_points_list

    def encode_attribute(self, points_list, colors_list, reconstructed_points_list, start_frame, early_return=False):
        exp_dir = self.configs['exp_dir']
        fix_start = self.configs['fix_start']
        fix_end = self.configs['fix_end']
        num_frames = len(points_list)
        model_dir = os.path.join(exp_dir, 'models')
        start_model = None
        end_model = None
        if num_frames == 1:
            suffix = f'{start_frame:04d}'
        else:
            end_frame = start_frame + num_frames - 1
            if fix_start:
                start_model = get_model('attribute', self.configs, self.device)
                start_model_path = os.path.join(model_dir, f'attr_model_{start_frame:04d}.pt')
                assert os.path.exists(start_model_path)
                start_model.load_state_dict(torch.load(start_model_path, map_location=self.device))
            if fix_end:
                end_model = get_model('attribute', self.configs, self.device)
                end_model_path = os.path.join(model_dir, f'attr_model_{end_frame:04d}.pt')
                assert os.path.exists(end_model_path)
                end_model.load_state_dict(torch.load(end_model_path, map_location=self.device))
            suffix = f'{start_frame:04d}-{end_frame:04d}'
        attr_model_path = os.path.join(model_dir, f'attr_model_{suffix}.pt')

        trainer = self.get_trainer('attribute', train=True, num_frames=num_frames,
                                   points_list=points_list, colors_list=colors_list, reconstructed_points_list=reconstructed_points_list,
                                   start_model=start_model, end_model=end_model)
        if os.path.exists(attr_model_path):
            self.logger.log('Load existing model...')
            trainer.model.load_state_dict(torch.load(attr_model_path, map_location=self.device))
        else:
            self.logger.log('Train model from scratch...')
            trainer.train(**get_train_kwargs('attribute', self.configs))
            torch.save(trainer.model.state_dict(), attr_model_path)
        if early_return:
            return
        self.logger.log('Encode and decode model...')
        model_bytes = trainer.model.encode()
        trainer.model.decode(model_bytes)
 
        bits = utils.bytes_to_bits(model_bytes)
        return bits
    
    def encode(self, encode_colors):
        dataset_dir = self.configs['dataset_dir']
        exp_dir = self.configs['exp_dir']
        sequence = self.configs['sequence']
        fix_start = self.configs['fix_start']
        fix_end = self.configs['fix_end']
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
            if fix_start:
                self.encode_geometry([points_list[start]], start_frame, encode_colors, early_return=True)
            if fix_end:
                self.encode_geometry([points_list[end - 1]], end_frame - 1, encode_colors, early_return=True)
            group_bits, reconstructed_points_list = self.encode_geometry(points_list[start:end], start_frame, encode_colors)
            num_group_bits = utils.digits_to_bits(np.array([len(group_bits)]), depth=32)
            bits_list.append(np.concatenate([num_group_bits, group_bits]))

            if encode_colors:
                self.logger.log('Encode attribute...')
                if fix_start:
                    self.encode_attribute([points_list[start]], [colors_list[start]], [reconstructed_points_list[0]], start_frame, early_return=True)
                if fix_end:
                    self.encode_attribute([points_list[end - 1]], [colors_list[end - 1]], [reconstructed_points_list[-1]], end_frame - 1, early_return=True)
                group_bits = self.encode_attribute(points_list[start:end], colors_list[start:end], reconstructed_points_list, start_frame)
                num_group_bits = utils.digits_to_bits(np.array([len(group_bits)]), depth=32)
                bits_list.append(np.concatenate([num_group_bits, group_bits]))
        
        bits = np.concatenate(bits_list)
        utils.write_binary(bits, bin_path)
        encode_time = time.time() - encode_time
        return encode_time
        
    def decode_geometry(self, bits, num_frames, start_frame):
        thresholds, num_blocks, blocks, model_bytes = self.split_bits(bits, num_frames)
        model = get_model('geometry', self.configs, self.device, curve=num_frames > 1)
        model.decode(model_bytes)

        if num_frames == 1:
            trainer = self.get_trainer('geometry', train=False, num_frames=num_frames, num_blocks=num_blocks, blocks=blocks, model=model)
            trainers = [trainer]
        else:
            trainers = []
            start = 0
            for frame_index in range(num_frames):
                model_t = get_model('geometry', self.configs, self.device)
                t = frame_index / (num_frames - 1)
                model.export_parameters_t(model_t, t)
                end = start + num_blocks[frame_index]
                blocks_t = blocks[start:end]
                start = end
                trainer_t = self.get_trainer('geometry', train=False, num_frames=1, num_blocks=np.array([num_blocks[frame_index]]), blocks=blocks_t, model=model_t)
                trainers.append(trainer_t)

        reconstructed_points_list = []
        for frame_index in range(num_frames):
            frame = start_frame + frame_index
            self.logger.log(f'Reconstruct frame {frame:04d} ({frame_index + 1}/{num_frames})...')
            reconstructed_points = trainers[frame_index].reconstruct_geometry(frame_index, thresholds[frame_index])
            reconstructed_points_list.append(reconstructed_points)
        return reconstructed_points_list
    
    def decode_attribute(self, reconstructed_points_list, bits, start_frame):
        num_frames = len(reconstructed_points_list)
        model_bytes = utils.bits_to_bytes(bits)
        model = get_model('attribute', self.configs, self.device, curve=num_frames > 1)
        model.decode(model_bytes)

        if num_frames == 1:
            trainer = self.get_trainer('attribute', train=False, num_frames=num_frames, reconstructed_points_list=reconstructed_points_list, model=model)
            trainers = [trainer]
        else:
            trainers = []
            for frame_index in range(num_frames):
                model_t = get_model('attribute', self.configs, self.device)
                t = frame_index / (num_frames - 1)
                model.export_parameters_t(model_t, t)
                trainer_t = self.get_trainer('attribute', train=False, num_frames=1, reconstructed_points_list=[reconstructed_points_list[frame_index]], model=model_t)
                trainers.append(trainer_t)

        reconstructed_colors_list = []
        for frame_index in range(num_frames):
            frame = start_frame + frame_index
            self.logger.log(f'Reconstruct frame {frame:04d} ({frame_index + 1}/{num_frames})...')
            reconstructed_colors = trainers[frame_index].reconstruct_attribute(frame_index)
            reconstructed_colors_list.append(reconstructed_colors)
        return reconstructed_colors_list
    
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
            reconstructed_points_list = self.decode_geometry(group_bits, end - start, start_frame)
            bits = bits[32 + num_group_bits:]

            if encode_colors:
                self.logger.log('Decode attribute...')
                num_group_bits = utils.bits_to_digits(bits[:32], depth=32).squeeze()
                group_bits = bits[32:32 + num_group_bits]
                reconstructed_colors_list = self.decode_attribute(reconstructed_points_list, group_bits, start_frame)
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