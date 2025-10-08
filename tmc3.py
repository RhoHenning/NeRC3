import os
import subprocess
import numpy as np
import utils

class TMC3:
    def __init__(self, tmc3_path):
        self.tmc3_path = tmc3_path

    @staticmethod
    def octree_raht_config(pqs, qp):
        return (
            f' --mode=0' +
            f' --trisoupNodeSizeLog2=0' +
            f' --mergeDuplicatedPoints=1' +
            f' --neighbourAvailBoundaryLog2=8' +
            f' --intra_pred_max_node_size_log2=6' +
            f' --positionQuantizationScale={pqs}' +
            f' --maxNumQtBtBeforeOt=4' +
            f' --minQtbtSizeLog2=0' +
            f' --planarEnabled=1' +
            f' --rahtPredictionSearchRange=50000' +
            f' --planarEnabled=0' +
            f' --planarModeIdcmUse=0' +
            f' --convertPlyColourspace=1' +
            f' --transformType=0' +
            f' --qp={qp}' +
            f' --qpChromaOffset=-2' +
            f' --bitdepth=8' +
            f' --attrOffset=0' +
            f' --attrScale=1' +
            f' --attribute=color'
        )

    @staticmethod
    def trisoup_raht_config(node_size_log2, qp):
        return (
            f' --mode=0' +
            f' --sliceMaxPointsTrisoup=1100000' +
            f' --sliceMaxPoints=1000000' +
            f' --sliceMinPoints=449000' +
            f' --neighbourAvailBoundaryLog2=8' +
            f' --intra_pred_max_node_size_log2=6' +
            f' --inferredDirectCodingMode=0' +
            f' --planarEnabled=1' +
            f' --rahtPredictionSearchRange=50000' +
            f' --planarEnabled=0' +
            f' --planarModeIdcmUse=0' +
            f' --positionQuantizationScale=1' +
            f' --trisoupNodeSizeLog2={node_size_log2}' +
            f' --trisoupQuantizationBits=2' +
            f' --trisoupCentroidResidualEnabled=1' +
            f' --trisoupFaceVertexEnabled=1' +
            f' --trisoupFineRayTracingEnabled=0' +
            f' --trisoupImprovedEncoderEnabled=1' +
            f' --convertPlyColourspace=1' +
            f' --transformType=0' +
            f' --qp={qp}' +
            f' --qpChromaOffset=-2' +
            f' --bitdepth=8' +
            f' --attribute=color'
        )

    def encode(self, cloud_path, bin_path, **kwargs):
        coding = kwargs['coding']
        pqs = kwargs['pqs']
        qp = kwargs.get('qp', None)
        node_size_log2 = kwargs.get('node_size_log2', None)
        encode_colors = kwargs['encode_colors']
        print_process = kwargs.get('print_process', False)

        assert coding in ['octree', 'trisoup']
        if coding == 'octree':
            coding_config = TMC3.octree_raht_config(pqs, qp)
        elif coding == 'trisoup':
            coding_config = TMC3.trisoup_raht_config(node_size_log2, qp)
        assert os.path.exists(cloud_path)
        config = (
            coding_config +
            f' --uncompressedDataPath={cloud_path}' +
            f' --compressedStreamPath={bin_path}' +
            f' --disableAttributeCoding={int(not encode_colors)}'
        )
        command = self.tmc3_path + config
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        encode_time = 0.0
        c = subp.stdout.readline()
        while c:
            line = c.decode(encoding='utf-8')
            if print_process:
                print(line, end='')
            words = ' '.join(line.split()).split()
            if 'Processing time (user):' in line:
                encode_time = float(words[3])
            c = subp.stdout.readline()
        return encode_time

    def decode(self, bin_path, reconstructed_path, print_process=False):
        assert os.path.exists(bin_path)
        config = (
            ' --mode=1' +
            ' --convertPlyColourspace=1' +
            f' --compressedStreamPath={bin_path}' +
            f' --reconstructedDataPath={reconstructed_path}' +
            ' --outputBinaryPly=0'
        )
        command = self.tmc3_path + config
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        decode_time = 0.0
        c = subp.stdout.readline()
        while c:
            line = c.decode(encoding='utf-8')
            if print_process:
                print(line, end='')
            words = ' '.join(line.split()).split()
            if 'Processing time (user):' in line:
                decode_time = float(words[3])
            c = subp.stdout.readline()
        # The attribute format of point clouds produced by tmc3 is GBR. We manually change it to RGB
        # This process is necessary when evaluating with PCQM
        with open(reconstructed_path, 'r') as file:
            lines = file.readlines()
        if len(lines) > 6 and lines[6] == 'property uchar green\n':
            lines[6] = 'property uchar red\n'
            lines[7] = 'property uchar green\n'
            lines[8] = 'property uchar blue\n'
            for index in range(12, len(lines)):
                words = ' '.join(lines[index].split()).split()
                words[3], words[4], words[5] = words[5], words[3], words[4]
                lines[index] = ' '.join(words) + '\n'
            with open(reconstructed_path, 'w') as file:
                for line in lines:
                    file.write(line)
        del lines
        return decode_time

class TMC3Coder:
    def __init__(self, logger, configs):
        self.logger = logger
        self.configs = configs
        tmc3_path = self.configs['tmc3_path']
        self.tmc3 = TMC3(tmc3_path)

    def encode(self, encode_colors):
        dataset_dir = self.configs['dataset_dir']
        exp_dir = self.configs['exp_dir']
        sequence = self.configs['sequence']
        global_num_frames = self.configs['num_frames']
        global_start_frame = self.configs['start_frame']
        kwargs = {
            'coding': self.configs['coding'],
            'pqs': self.configs['pqs'],
            'qp': self.configs['qp'],
            'node_size_log2': self.configs['node_size_log2'],
            'encode_colors': encode_colors,
            'print_process': False
        }
        bin_path = os.path.join(exp_dir, 'encoded.bin')
        temp_bin_path = os.path.join(exp_dir, 'temp.bin')
        
        encode_time = 0.0
        bits_list = []
        for frame_index in range(global_num_frames):
            frame = global_start_frame + frame_index
            self.logger.log(f'Encode frame {frame:04d} ({frame_index + 1}/{global_num_frames})...')
            cloud_path = os.path.join(dataset_dir, f'{sequence}_{frame:04d}.ply')
            encode_time += self.tmc3.encode(cloud_path, temp_bin_path, **kwargs)
            frame_bits = utils.load_binary(temp_bin_path)
            num_frame_bits = utils.digits_to_bits(np.array([len(frame_bits)]), depth=32)
            bits_list.append(np.concatenate([num_frame_bits, frame_bits]))
        bits = np.concatenate(bits_list)
        utils.write_binary(bits, bin_path)
        os.remove(temp_bin_path)
        return encode_time
        
    def decode(self, encode_colors):
        exp_dir = self.configs['exp_dir']
        global_num_frames = self.configs['num_frames']
        global_start_frame = self.configs['start_frame']
        reconstructed_dir = os.path.join(exp_dir, 'reconstructed')
        bin_path = os.path.join(exp_dir, 'encoded.bin')
        temp_bin_path = os.path.join(exp_dir, 'temp.bin')
        assert os.path.exists(bin_path)

        decode_time = 0.0
        bits = utils.load_binary(bin_path)
        for frame_index in range(global_num_frames):
            frame = global_start_frame + frame_index
            self.logger.log(f'Decode frame {frame:04d} ({frame_index + 1}/{global_num_frames})...')
            num_frame_bits = utils.bits_to_digits(bits[:32], depth=32).squeeze()
            frame_bits = bits[32:32 + num_frame_bits]
            utils.write_binary(frame_bits, temp_bin_path)
            reconstructed_path = os.path.join(reconstructed_dir, f'reconstructed_{frame:04d}.ply')
            decode_time += self.tmc3.decode(temp_bin_path, reconstructed_path, print_process=False)
            bits = bits[32 + num_frame_bits:]
        os.remove(temp_bin_path)
        return decode_time