import os
import argparse
import yaml
import torch
from collections import defaultdict
import utils
from logger import Logger
from tmc2 import TMC2Coder
from tmc3 import TMC3Coder
from coder import Coder
from coder_i import CoderCurve
from evaluator import Evaluator

import warnings
warnings.filterwarnings('ignore')

def get_configs():
    _config_path = f'_{os.getpid()}.yaml'
    os.system(f'cp {args.config_path} {_config_path}')
    if args.modification is None:
        args.modification = []
    modification_dict = {}
    for item_str in args.modification:
        item = item_str.split('=')
        modification_dict[item[0]] = item[1]
    with open(_config_path, 'r') as file:
        lines = file.readlines()
    flag = None
    with open(_config_path, 'w') as file:
        for line in lines:
            if ':' in line:
                name = line.split(':')[0]
                if name in ['geometry', 'attribute']:
                    flag = name
                else:
                    if name.lstrip() != name:
                        assert flag is not None
                        key = f'{flag}.{name.lstrip()}'
                    else:
                        key = name
                        flag = None
                    if key in modification_dict:
                        line = f'{name}: {modification_dict[key]}\n'
            file.write(line)
    del lines
    with open(_config_path) as file:
        configs = yaml.safe_load(file)
    exp_dir = configs['exp_dir']
    config_path = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(config_path):
        print(f'Load existing configuration file: {config_path}')
        os.system(f'rm {_config_path}')
        with open(config_path) as file:
            configs = yaml.safe_load(file)
    else:
        os.makedirs(exp_dir, exist_ok=True)
        os.system(f'mv {_config_path} {config_path}')
    return configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument('--config_path', type=str, default='config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--encode_colors', action='store_true')
    parser.add_argument('--modification', nargs='*', type=str)
    args = parser.parse_args()
    configs = get_configs()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    dataset_dir = configs['dataset_dir']
    exp_dir = configs['exp_dir']
    method = configs['method']
    sequence = configs['sequence']
    num_frames = configs['num_frames']
    start_frame = configs['start_frame']
    depth = configs['depth']
    pc_error_path = configs['pc_error_path']
    pcqm_dir = configs['pcqm_dir']
    
    config_path = os.path.join(exp_dir, 'config.yaml')
    log_path = os.path.join(exp_dir, 'log.txt')
    result_path = os.path.join(exp_dir, 'result.csv')
    reconstructed_dir = os.path.join(exp_dir, 'reconstructed')
    bin_path = os.path.join(exp_dir, 'encoded.bin')
    os.makedirs(reconstructed_dir, exist_ok=True)
    if method in ['nerc3', 'cnerc3']:
        model_dir = os.path.join(exp_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
    pyname = os.path.basename(__file__)
    logger = Logger(pyname, log_path)
    logger.log('*' * 40 + f' {pyname} ' + '*' * 40)
    
    assert method in ['nerc3', 'cnerc3', 'tmc2', 'tmc3']
    if method == 'nerc3':
        coder = Coder(logger, configs, device)
    elif method == 'cnerc3':
        coder = CoderCurve(logger, configs, device)
    elif method == 'tmc2':
        coder = TMC2Coder(logger, configs)
    elif method == 'tmc3':
        coder = TMC3Coder(logger, configs)
    encode_time = coder.encode(args.encode_colors)
    logger.log(f'Encoding time: {encode_time / num_frames:.6} s/frame')
    decode_time = coder.decode(args.encode_colors)
    logger.log(f'Decoding time: {decode_time / num_frames:.6} s/frame')

    sum_points = 0
    cloud_names = []
    for frame_index in range(num_frames):
        frame = start_frame + frame_index
        cloud_names.append(f'{sequence}_{frame:04d}.ply')
    count_path = os.path.join(dataset_dir, 'counts.txt')
    with open(count_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        words = line.strip().split()
        if len(words) == 2 and words[1] in cloud_names:
            sum_points += int(words[0])

    if True:
        bits = utils.load_binary(bin_path)
        logger.log(f'Compressed size: {len(bits)} bits ({len(bits) / sum_points:.6} bpp)')
        evaluator = Evaluator(pc_error_path, pcqm_dir)
        sum_distortion = defaultdict(float)
        for frame_index in range(num_frames):
            frame = start_frame + frame_index
            logger.log(f'Evaluate frame {frame:04d} ({frame_index + 1}/{num_frames})...')
            cloud_path = os.path.join(dataset_dir, f'{sequence}_{frame:04d}.ply')
            reconstructed_path = os.path.join(reconstructed_dir, f'reconstructed_{frame:04d}.ply')
            pc_error_results = evaluator.pc_error(cloud_path, reconstructed_path, (1 << depth) - 1, has_colors=args.encode_colors)
            for metric in ['D1', 'D2', 'Y', 'YUV']:
                metric_psnr = f'{metric} PSNR'
                if metric in pc_error_results and metric_psnr in pc_error_results:
                    logger.log(f'{metric}: {pc_error_results[metric]:.6} ' +
                               f'PSNR: {pc_error_results[metric_psnr]:.6}')
                    sum_distortion[metric_psnr] += pc_error_results[metric_psnr]
            if args.encode_colors:
                pcqm_result = evaluator.pcqm(cloud_path, reconstructed_path)
                logger.log(f'PCQM: {pcqm_result:.6}')
                sum_distortion['PCQM'] += pcqm_result
        for metric in sum_distortion.keys():
            logger.log(f'Average {metric}: {sum_distortion[metric] / num_frames:.6}')