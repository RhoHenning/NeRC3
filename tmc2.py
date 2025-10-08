import os
import subprocess

class TMC2:
    sequence_config = (
        ' --geometry3dCoordinatesBitdepth=10' +
        ' --geometryNominal2dBitdepth=8' +
        ' --roiBoundingBoxMinX=0,0,0,0' +
        ' --roiBoundingBoxMaxX=1023,1023,1023,1023' +
        ' --roiBoundingBoxMinY=0,256,512,768' +
        ' --roiBoundingBoxMaxY=255,511,767,1023' +
        ' --roiBoundingBoxMinZ=0,0,0,0' +
        ' --roiBoundingBoxMaxZ=1023,1023,1023,1023' +
        ' --numTilesHor=2' +
        ' --tileHeightToWidthRatio=1' +
        ' --numCutsAlong1stLongestAxis=2' +
        ' --numCutsAlong2ndLongestAxis=1' +
        ' --numCutsAlong3rdLongestAxis=1'
    )
    
    def __init__(self, tmc2_bin_dir, tmc2_config_dir, hdrtools_bin_dir):
        self.tmc2_bin_dir = tmc2_bin_dir
        self.tmc2_config_dir = tmc2_config_dir
        self.hdrtools_bin_dir = hdrtools_bin_dir
        self.encoder_path = os.path.join(self.tmc2_bin_dir, 'PccAppEncoder')
        self.decoder_path = os.path.join(self.tmc2_bin_dir, 'PccAppDecoder')
        self.hdrconvert_path = os.path.join(self.hdrtools_bin_dir, 'HDRConvert')

        assert os.path.exists(self.encoder_path)
        assert os.path.exists(self.decoder_path)
        assert os.path.exists(self.hdrconvert_path)

    def encode(self, cloud_dir, bin_path, **kwargs):
        condition = kwargs['condition']
        sequence = kwargs['sequence']
        geom_qp = kwargs['geom_qp']
        attr_qp = kwargs['attr_qp']
        num_frames = kwargs['num_frames']
        start_frame = kwargs['start_frame']
        group_size = kwargs['group_size']
        encode_colors = kwargs['encode_colors']
        print_process = kwargs.get('print_process', False)

        common_config_path = os.path.join(self.tmc2_config_dir, 'common', 'ctc-common.cfg')
        assert condition in ['all-intra', 'low-delay', 'random-access']
        condition_config_path = os.path.join(self.tmc2_config_dir, 'condition', f'ctc-{condition}.cfg')
        assert os.path.exists(common_config_path)
        assert os.path.exists(condition_config_path)
        
        sequence_config_path = os.path.join(self.tmc2_config_dir, 'sequence', f'{sequence}.cfg')
        if os.path.exists(sequence_config_path):
            sequence_config = f' --config={sequence_config_path}'
        else:
            sequence_config = TMC2.sequence_config
        cloud_path = os.path.join(cloud_dir, sequence + r'_%04d.ply')
        config = (
            f' --configurationFolder={self.tmc2_config_dir}' +
            f' --config={common_config_path}' +
            f' --config={condition_config_path}' +
            sequence_config +
            f' --geometryQP={geom_qp}' +
            f' --attributeQP={attr_qp}' +
            f' --occupancyPrecision=4' +
            f' --colorTransform=0' +
            f' --uncompressedDataPath={cloud_path}' +
            f' --colorSpaceConversionPath={self.hdrconvert_path}' +
            f' --compressedStreamPath={bin_path}' +
            f' --frameCount={num_frames}' +
            f' --startFrameNumber={start_frame}' +
            f' --groupOfFramesSize={group_size}' +
            f' --noAttributes={int(not encode_colors)}'
        )
        command = self.encoder_path + config
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        encode_time = 0.0
        c = subp.stdout.readline()
        while c:
            line = c.decode(encoding='utf-8')
            if print_process:
                print(line, end='')
            words = ' '.join(line.split()).split()
            if 'Processing time (user.self):' in line:
                encode_time = float(words[3])
            c = subp.stdout.readline()
        return encode_time

    def decode(self, bin_path, reconstructed_dir, start_frame, print_process=False):
        reconstructed_path = os.path.join(reconstructed_dir, r'reconstructed_%04d.ply')
        hdrconvert_config_path = os.path.join(self.tmc2_config_dir, 'hdrconvert', 'yuv420torgb444.cfg')
        assert os.path.exists(bin_path)
        assert os.path.exists(hdrconvert_config_path)
        config = (
            f' --startFrameNumber={start_frame}' +
            f' --compressedStreamPath={bin_path}' +
            f' --reconstructedDataPath={reconstructed_path}' +
            f' --colorSpaceConversionPath={self.hdrconvert_path}' +
            f' --inverseColorSpaceConversionConfig={hdrconvert_config_path}' +
            f' --nbThread=1' +
            f' --colorTransform=0'
        )
        command = self.decoder_path + config
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        decode_time = 0.0
        c = subp.stdout.readline()
        while c:
            line = c.decode(encoding='utf-8')
            if print_process:
                print(line, end='')
            words = ' '.join(line.split()).split()
            if 'Processing time (user.self):' in line:
                decode_time = float(words[3])
            c = subp.stdout.readline()
        return decode_time

class TMC2Coder:
    def __init__(self, logger, configs):
        self.logger = logger
        self.configs = configs
        tmc2_bin_dir = self.configs['tmc2_bin_dir']
        tmc2_config_dir = self.configs['tmc2_config_dir']
        hdrtools_bin_dir = self.configs['hdrtools_bin_dir']
        self.tmc2 = TMC2(tmc2_bin_dir, tmc2_config_dir, hdrtools_bin_dir)

    def encode(self, encode_colors):
        dataset_dir = self.configs['dataset_dir']
        exp_dir = self.configs['exp_dir']
        kwargs = {
            'condition': self.configs['condition'],
            'sequence': self.configs['sequence'],
            'geom_qp': self.configs['geom_qp'],
            'attr_qp': self.configs['attr_qp'],
            'num_frames': self.configs['num_frames'],
            'start_frame': self.configs['start_frame'],
            'group_size': self.configs['group_size'],
            'encode_colors': encode_colors,
            'print_process': False
        }
        bin_path = os.path.join(exp_dir, 'encoded.bin')
        self.logger.log(f'Encode frames...')
        encode_time = self.tmc2.encode(dataset_dir, bin_path, **kwargs)
        return encode_time
        
    def decode(self, encode_colors):
        exp_dir = self.configs['exp_dir']
        start_frame = self.configs['start_frame']
        reconstructed_dir = os.path.join(exp_dir, 'reconstructed')
        bin_path = os.path.join(exp_dir, 'encoded.bin')
        assert os.path.exists(bin_path)
        self.logger.log(f'Decode frames...')
        decode_time = self.tmc2.decode(bin_path, reconstructed_dir, start_frame, print_process=False)
        return decode_time