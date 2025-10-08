import os
import utils

if __name__ == '__main__':
    for sequence in ['longdress', 'loot', 'redandblack', 'soldier']:
        start_frame = {'longdress': 1051, 'loot': 1000, 'redandblack': 1450, 'soldier': 536}[sequence]
        raw_cloud_dir = os.path.join('./8iVFBv2', sequence, 'Ply')
        cloud_dir = os.path.join('./dataset', '8iVFB', sequence)
        os.makedirs(cloud_dir, exist_ok=True)
        lines = []
        for frame_index in range(32):
            frame = start_frame + frame_index
            raw_cloud_path = os.path.join(raw_cloud_dir, f'{sequence}_vox10_{frame:04d}.ply')
            assert os.path.exists(raw_cloud_path)
            cloud_path = os.path.join(cloud_dir, f'{sequence}_vox10_{frame:04d}.ply')
            print(cloud_path)
            points, colors, _ = utils.load_ply_cloud(raw_cloud_path)
            points, colors = utils.convert_bit_depth(points, colors, 10)
            normals = utils.estimate_normals(points)
            utils.write_ply_cloud(cloud_path, points, colors, normals)
            lines.append(f'{len(points)} {os.path.basename(cloud_path)}\n')
        count_path = os.path.join(cloud_dir, 'counts.txt')
        with open(count_path, 'w') as file:
            for line in lines:
                file.write(line)

    for sequence in ['basketball_player', 'dancer', 'exercise', 'model']:
        start_frame = 1
        raw_cloud_dir = os.path.join('./Owlii', f'{sequence}_vox11')
        cloud_dir = os.path.join('./dataset', 'Owlii', sequence)
        os.makedirs(cloud_dir, exist_ok=True)
        lines = []
        for frame_index in range(32):
            frame = start_frame + frame_index
            raw_cloud_path = os.path.join(raw_cloud_dir, f'{sequence}_vox11_{frame:08d}.ply')
            assert os.path.exists(raw_cloud_path)
            cloud_path = os.path.join(cloud_dir, f'{sequence}_vox10_{frame:04d}.ply')
            print(cloud_path)
            points, colors, _ = utils.load_ply_cloud(raw_cloud_path)
            points, colors = utils.convert_bit_depth(points, colors, 11)
            normals = utils.estimate_normals(points)
            utils.write_ply_cloud(cloud_path, points, colors, normals)
            lines.append(f'{len(points)} {os.path.basename(cloud_path)}\n')
        count_path = os.path.join(cloud_dir, 'counts.txt')
        with open(count_path, 'w') as file:
            for line in lines:
                file.write(line)
