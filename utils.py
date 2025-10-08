import numpy as np
import open3d as o3d

def bits_to_digits(bits, depth=8):
    return np.sum(bits.astype(np.int64).reshape(-1, depth) << np.flip(np.arange(depth)), axis=-1)

def bits_to_bytes(bits):
    return bits_to_digits(bits).astype(np.int8).tobytes()

def digits_to_bits(digits, depth=8):
    return (digits.repeat(depth) >> np.tile(np.flip(np.arange(depth)), len(digits))) & 1

def bytes_to_bits(byts):
    return digits_to_bits(np.frombuffer(byts, dtype=np.int8))

def load_binary(path):
    with open(path, 'rb') as file:
        data = bytes_to_bits(file.read())
    return data

def write_binary(data, path):
    with open(path, 'wb') as file:
        file.write(bits_to_bytes(data))

def load_ply_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.array(pcd.points).astype(np.float32)
    colors, normals = None, None
    if pcd.has_colors():
        colors = np.array(pcd.colors).astype(np.float32)
    if pcd.has_normals():
        normals = np.array(pcd.normals).astype(np.float32)
    return points, colors, normals

def write_ply_cloud(path, points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)
    with open(path, 'r') as file:
        cloud = file.read()
    with open(path, 'w') as file:
        file.write(cloud.replace('double', 'float'))
    del cloud

def convert_bit_depth(points, colors, raw_depth, depth=10):
    assert raw_depth >= depth
    if raw_depth == depth:
        return points, colors
    scale = 1 << (raw_depth - depth)
    new_points, indices, counts = np.unique(points // scale, axis=0, return_inverse=True, return_counts=True)
    new_colors = np.zeros_like(new_points)
    np.add.at(new_colors, indices, colors)
    new_colors = new_colors / counts[..., np.newaxis]
    return new_points, new_colors

def partition_blocks(points, depth, block_depth):
    block_occupancy = np.zeros((1 << block_depth,) * 3)
    block_width = 1 << (depth - block_depth)
    for point in points:
        block = divmod(point.astype(np.int64), block_width)[0]
        block_occupancy[tuple(block)] = 1
    blocks = np.argwhere(block_occupancy).astype(np.float32)
    return blocks

def normalize(points, min_bound, max_bound):
    assert (points >= min_bound).all() and (points <= max_bound).all()
    points = (points - min_bound) / (max_bound - min_bound)
    points = points * 2 - 1
    return points

def denormalize(points, min_bound, max_bound):
    points = points / 2 + 0.5
    points = points * (max_bound - min_bound) + min_bound
    return points

def estimate_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    normals = np.array(pcd.normals).astype(np.float32)
    return normals

def nearest_neighbor_indices(points, voxels, knn=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    indices = np.zeros((len(voxels), knn), dtype=np.int64)
    for i, voxel in enumerate(voxels):
        indices[i] = kd_tree.search_knn_vector_3d(voxel, knn)[1]
    return indices

def chamfer_distance(points_1, points_2):
    indices_12 = nearest_neighbor_indices(points_2, points_1).squeeze(axis=-1)
    indices_21 = nearest_neighbor_indices(points_1, points_2).squeeze(axis=-1)
    dist_12 = np.sum((points_1 - points_2[indices_12]) ** 2, axis=-1)
    dist_21 = np.sum((points_2 - points_1[indices_21]) ** 2, axis=-1)
    dist = np.maximum(np.mean(dist_12), np.mean(dist_21))
    return dist