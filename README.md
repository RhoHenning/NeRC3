# NeRC<sup>3</sup>: Implicit Neural Representations for Colored Point Cloud Compression

Official implementation of paper [Implicit Neural Compression of Point Clouds](https://ieeexplore.ieee.org/document/11322701), accepted by IEEE Transactions on Image Processing.

## Requirements

- pytorch 1.9.0
- open3d 0.18.0
- deepCABAC 0.1.0: https://github.com/fraunhoferhhi/DeepCABAC
- pc_error: https://github.com/minhkstn/mpeg-pcc-dmetric
- PCQM: https://github.com/MEPP-team/PCQM

Test datasets:

- 8iVFB dataset: https://plenodb.jpeg.org/pc/8ilabs
- Owlii dataset: https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/

Baselines:

- tmc3 v23.0-rc2: https://github.com/MPEGGroup/mpeg-pcc-tmc13
- tmc2 v25.0: https://github.com/MPEGGroup/mpeg-pcc-tmc2
- HDRTools v0.24: https://gitlab.com/standards/HDRTools

## Example Usage

Process raw datasets by extracting the first 32 frames of each sequence, downsampling points to 10 bits, and estimating normals for D2 calculation.

```shell
python prepare_dataset.py
```

### Proposed schemes

Compress the first frame of *longdress* by i-NeRC<sup>3</sup>.

```shell
# i-NeRC3
python main.py --config_path=./config.yaml --encode_colors --modification exp_dir=./exp/nerc3/longdress/l1t1f1 \
    method=nerc3 geometry.lmbda=1.0 attribute.lmbda=1.0 group_size=1 num_frames=1 \
    dataset_dir=./dataset/8iVFB/longdress sequence=longdress_vox10 start_frame=1051
```

Compress the first 8 frames of *longdress* by r-NeRC<sup>3</sup>.

```shell
# r-NeRC3
python main.py --config_path=./config.yaml --encode_colors --modification exp_dir=./exp/nerc3/longdress/l1t1f8d \
    method=nerc3 geometry.lmbda=1.0 attribute.lmbda=1.0 group_size=1 num_frames=8 encode_diff=True \
    dataset_dir=./dataset/8iVFB/longdress sequence=longdress_vox10 start_frame=1051
```

Compress the first 8 frames of *soldier* by c-NeRC<sup>3</sup>.

```shell
# c-NeRC3
python main.py --config_path=./config.yaml --encode_colors --modification exp_dir=./exp/cnerc3/soldier/n3l1t8f8 \
    method=cnerc3 geometry.lmbda=1.0 attribute.lmbda=1.0 group_size=8 num_frames=8 \
    dataset_dir=./dataset/8iVFB/soldier sequence=soldier_vox10 start_frame=536
```

Compress the first 8 frames of *soldier* by 4D-NeRC<sup>3</sup>.

```shell
# 4D-NeRC3
python main.py --config_path=./config.yaml --encode_colors --modification exp_dir=./exp/nerc3/soldier/l1t8f8 \
    method=nerc3 geometry.lmbda=1.0 attribute.lmbda=1.0 group_size=8 num_frames=8 \
    dataset_dir=./dataset/8iVFB/soldier sequence=soldier_vox10 start_frame=536
```

### Baselines

Compress the first frame of *basketball\_player* by G-PCC/V-PCC.

```shell
# G-PCC (octree)
python main.py --config_path=./config.yaml --encode_colors --modification exp_dir=./exp/tmc3/basketball_player/p0.25q46f1 \
    method=tmc3 coding=octree pqs=0.25 qp=46 num_frames=1 \
    dataset_dir=./dataset/Owlii/basketball_player sequence=basketball_player_vox10 start_frame=1
# G-PCC (trisoup)
python main.py --config_path=./config.yaml --encode_colors --modification exp_dir=./exp/tmc3/basketball_player/g2q22f1 \
    method=tmc3 coding=trisoup node_size_log2=2 qp=22 num_frames=1 \
    dataset_dir=./dataset/Owlii/basketball_player sequence=basketball_player_vox10 start_frame=1
# V-PCC
python main.py --config_path=./config.yaml --encode_colors --modification exp_dir=./exp/tmc2/basketball_player/q32q42t1f1 \
    method=tmc2 condition=all-intra geom_qp=32 attr_qp=42 num_frames=1 group_size=1 \
    dataset_dir=./dataset/Owlii/basketball_player sequence=basketball_player_vox10 start_frame=1
```

## Citation

If you find this work helpful in your research, please cite as follows:

```
@article{ruan2025nerc3,
  title={Implicit Neural Compression of Point Clouds},
  author={Ruan, Hongning and Shao, Yulin and Yang, Qianqian and Zhao, Liang and Zhang, Zhaoyang and Niyato, Dusit},
  journal={IEEE Transactions on Image Processing},
  volume={35},
  number={},
  pages={260-275},
  year={2026},
  doi={10.1109/TIP.2025.3648141}
}
```