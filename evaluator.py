import os
import subprocess
import numpy as np

class Evaluator:
    metrics = {
        'mseF(p2point):'     : 'D1',
        'mseF,PSNR(p2point):': 'D1 PSNR',
        'mseF(p2plane):'     : 'D2',
        'mseF,PSNR(p2plane):': 'D2 PSNR',
        'c[0],F': 'Y',
        'c[1],F': 'U',
        'c[2],F': 'V',
        'c[0],PSNRF:': 'Y PSNR',
        'c[1],PSNRF:': 'U PSNR',
        'c[2],PSNRF:': 'V PSNR'
    }

    def __init__(self, pc_error_path, pcqm_dir):
        self.pc_error_path = pc_error_path
        self.pcqm_dir = pcqm_dir
        self.pcqm_path = os.path.join(self.pcqm_dir, 'PCQM')
        assert os.path.exists(self.pc_error_path)
        assert os.path.exists(self.pcqm_path)

    @staticmethod
    def mean_psnr(y_psnr, u_psnr, v_psnr):
        return -10 * np.log10((np.power(10, -y_psnr / 10) * 6 + np.power(10, -u_psnr / 10) + np.power(10, -v_psnr / 10)) / 8)

    def pc_error(self, cloud_path_1, cloud_path_2, resolution, has_colors=True, print_process=False):
        config = (
            f' --fileA={cloud_path_1}' +
            f' --fileB={cloud_path_2}' +
            f' --color={int(has_colors)}' +
            f' --resolution={resolution}' +
            f' --dropdups=2' +
            f' --neighborsProc=1'
        )
        command = self.pc_error_path + config
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        results = {}
        c = subp.stdout.readline()
        while c:
            line = c.decode(encoding='utf-8')
            if print_process:
                print(line, end='')
            words = ' '.join(line.split()).split()
            if len(words) >= 3 and words[0] + words[1] in Evaluator.metrics:
                results[Evaluator.metrics[words[0] + words[1]]] = float(words[-1])
            c = subp.stdout.readline()
        if has_colors:
            results['YUV'] = (results['Y'] * 6 + results['U'] + results['V']) / 8
            results['YUV PSNR'] = Evaluator.mean_psnr(results['Y PSNR'], results['U PSNR'], results['V PSNR'])
        return results
    
    def pcqm(self, cloud_path_1, cloud_path_2, print_process=False):
        cloud_path_1 = os.path.abspath(cloud_path_1)
        cloud_path_2 = os.path.abspath(cloud_path_2)
        cwd = os.getcwd()
        os.chdir(self.pcqm_dir)
        command = f'./PCQM {cloud_path_1} {cloud_path_2} -r 0.004 -knn 20 -rx 2.0 -fq'
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        c = subp.stdout.readline()
        result = None
        while c:
            line = c.decode(encoding='utf-8')
            if print_process:
                print(line, end='')
            words = ' '.join(line.split()).split()
            if words[0] == 'PCQM':
                result = float(words[-1])
            c = subp.stdout.readline()
        os.chdir(cwd)
        return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument('--pc_error_path', type=str, default='./pc_error')
    parser.add_argument('--pcqm_dir', type=str, default='/home/ubuntu/PCQM/build/')
    parser.add_argument('--has_colors', action='store_true')
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--cloud_path', type=str)
    parser.add_argument('--reconstructed_path', type=str)
    args = parser.parse_args()
    
    evaluator = Evaluator(args.pc_error_path, args.pcqm_dir)
    pc_error_results = evaluator.pc_error(args.cloud_path, args.reconstructed_path, (1 << args.depth) - 1, has_colors=args.has_colors)
    for metric in ['D1', 'D2', 'Y', 'YUV']:
        metric_psnr = f'{metric} PSNR'
        if metric in pc_error_results and metric_psnr in pc_error_results:
            print(f'{metric}: {pc_error_results[metric]:.6} ' +
                   f'PSNR: {pc_error_results[metric_psnr]:.6}')
    if args.has_colors:
        pcqm_result = evaluator.pcqm(args.cloud_path, args.reconstructed_path)
        print(f'PCQM: {pcqm_result:.6}')