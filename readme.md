# SeMLaPS: Real-time Semantic Mapping with Latent Prior Networks and Quasi-Planar Segmentation
by Jingwen Wang, Juan Tarrio, Lourdes Agapito, Pablo F. Alcantarilla, Alexander Vakhitov.

Jingwen Wang (jingwen.wang.17@ucl.ac.uk) is the original author of the core of the method and the evaluaton scripts.
Alexander Vakhitov (alexander@slamcore.com) is the author of the QPOS over-segmentation method implementation.

This repository contains the code to train and evaluate the method and the link to the Semantic Mapping with Realsense dataset.


## 1. Set up environment

python=3.9
pytorch=1.11.0
cuda=11.3

```bash
conda env create -f environment.yml
conda activate semlaps
```

pytorch3d
```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
```

## 2. Data Preparation

### 2.1 ScanNet Data

You can download ScanNet by following their official instruction. 
Apart from the basic data, you will also need 2D semantic GT and 3D meshes with GT labels. You will expect to have the following fil types:

* `.sens`: extracted to `depth`, `pose`, `color`, `intrinsic`
* `.txt`: some meta-data
* `_vh_clean_2.labels.ply`: 3D GT mesh
* `_2d-label-filt.zip`: 2D semantic data

You will also need to process the raw 2D semantic images, resizing them to `640x480` and convert the semantic encoding from NYU-40 to ScanNt-20.

Unfortunately, we are not able to provide the code for this part. Please refer to official ScanNet GitHub or raise an issue if you have questions.


### 2.2 SMR dataset
Please find the Semantic Mapping for Realsense dataset [here] (https://drive.google.com/file/d/1_jHFrrQj6_9JK6G4NkTSoWSDiLESeE-q/view?usp=drive_link). 

## 3. Train LPN

### First, Create multi-view frames data

```bash
python create_fragments_n_views.py --scannet_root ${scannet_root} --save_files_root image_pairs
```

This will creates the multi-view training indices (triplet) of the camera frames for all the 1513 scenes in the train/val dataset of ScanNet.

### Train LPN with multi-view latent feature fusion

Training:

```bash
python train_lpn.py --config configs/config_lpn.yaml --scannet_root /media/jingwen/Data2/scannet/scans --log_dir exps/LPN
```

LPN supports 4 different modes for you to explore:

1. Multi-view RGBD (default): rgb and depth fusion with SSMA + feature warping (w/ depth, camera poses and K) `modality=rgbd, use_ssma=True, reproject=True`
2. Multi-view RGBD with RGB feature: rgb-only encoder, no SSMA, depth only used in feature warping (w/ depth, camera poses and K) `modality=rgbd, use_ssma=False, reproject=True`
3. Single-view RGBD: rgb and depth fusion with SSMA `modality=rgbd, use_ssma=True, reproject=False`
4. Single-view RGB: rgb input only `modality=rgb, use_ssma=False, reproject=False`

Evaluate script for LPN (2D):

```bash
python eval_lpn.py --log_dir exps/LPN --dataset_type scannet --dataset_root /media/jingwen/Data3/scannet/scans --save_dir exps/LPN/eval/scannet_val --eval
```

## 4. Train SegConvNet

### Step 1: Run offline QPOS and get segments

```bash
segment_suffix=segments/QPOS
python run_qpos.py --segment_suffix ${segment_suffix} --dataset_type scannet --dataset_root ${scannet_root} --small_segment_size 30 --expected_segment_size 60
```
Segments will be saved under `${scannet_root}/${scene}/${segment_suffix}` for each scene. Note that for slamcore sequences we have to adjust the segment size `--small_segment_size 120 --expected_segment_size 240`

### Step 2: Label GT meshes with BayesianFusion and LPN inference results

```bash
log_dir=logs/LPN
label_fusion_dir=exps/LPN_labels_3D
python eval_lpn_bayesian_label.py --log_dir ${log_dir} --dataset_type scannet --dataset_root ${scannet_root} --save_dir ${label_fusion_dir}
```

Labelled meshes will be saved under `${label_fusion_dir}/${scene}`

### Step 3: Prepare the training data

```bash
python prepare_3d_training_data.py --label_fusion_dir ${label_fusion_dir} --segment_suffix ${segment_suffix} --dataset_type scannet --dataset_root ${scannet_root} --save_mesh
```
The training data will be saved under `${label_fusion_dir}/${scene}/${segment_suffix}`

### Step 4: Train SegConvNet

```bash
python train_segconvnet.py --config configs/config_segconvnet.yaml  --log_dir exps/SegConvNet --label_fusion_dir ${label_fusion_dir} --segment_suffix ${segment_suffix}
```

Evaluation script for the SegConvNet:

```bash
segconv_logdir=exps/SegConvNet
python eval_segconvnet.py --log_dir ${segconv_logdir} --dataset_type scannet --dataset_root ${scannet_root} --label_fusion_dir ${label_fusion_dir} --segment_suffix ${segment_suffix} --save_dir exps/SegConvNet_labels
```

To reproduce results on the SMR dataset from the paper, please do steps 1, 2, 3 and then run the evaluation script for the SegConvNet eval_segconvnet.py.
## 5. Sequential Inference

### Run ScanNet sequential simulator

First download example scannet `scene0645_00` from [here](https://drive.google.com/file/d/1VRDydi0OVoXVVH-05EnibavFEByQH9a-/view?usp=drive_link) and extract it under `$SCANNET_ROOT`. You then should expect to have the following directory structure:
```
$SCANNET_ROOT
├── scene0645_00
    ├── color
        ├── 0.jpg
        ├── 1.jpg
        ...
    ├── depth
        ├── 0.png
        ├── 1.png
        ...
    ├── intrinsic
    ├── pose
    scene0645_00_vh_clean_2.ply
...
```
Then you need to update the ScanNet root path mapping [here](https://github.com/JingwenWang0226/differentiable_slam_map/blob/master/config.py#L8:L15), simply put your hostname and `$SCANNET_ROOT` as the key and value. 

Then you also need to download the checkpoint files:
- LPN: https://drive.google.com/file/d/1kz_5DVowhN06TH3yflCN7UQCkS1FCHMi/view?usp=drive_link
- LPN_rgb: https://drive.google.com/file/d/1E7e_Prhyq4iH09n99_ZhBwV4rSF3mYhC/view?usp=drive_link
- SegConvNet: https://drive.google.com/file/d/1FN8CYwOD4lB1gIto3KLSEuKKkWH29XM0/view?usp=drive_link

And extract them under `$EXP_DIR`. Then run the following command:
```
python sequential_runner_scannet.py --exp_dir $EXP_DIR --scene scene0645_00 --mapping_every 20 --skip 1
```
This will save the results under `$EXP_DIR/scannet/scene0645_00_skip20`

### Run RealSense sequential simulator
First download example RealSense sequence`kitchen1` from [here](https://drive.google.com/file/d/1v_1qDDKSVuMRtGYM8zIprR8xGKgPXMGU/view?usp=drive_link) and extract it under `$SLAMCORE_ROOT`. You then should expect to have the following directory structure:
```
$SLAMCORE_ROOT
├── kitchen1
    ├── color
        ├── 0.png
        ├── 10.png
        ├── 20.png
        ...
    ├── depth
        ├── 0.png
        ├── 10.png
        ├── 20.png
        ...
    ├── pose
        ├── 0.txt
        ├── 10.txt
        ├── 20.txt
    align.txt
    K.txt
    global_map_mesh.clean.ply
...
```
Then you need to update the SMR root path mapping [here](https://github.com/JingwenWang0226/differentiable_slam_map/blob/master/config.py#L8:L15), 
Note that `align.txt` is a transformation matrix (translation only) to shift the origin to approximately `np.min(verts, axis=0)`. You can simply save it when creating the mesh.

And extract them under `$EXP_DIR`. Then run the following command:
```
python sequential_runner_slamcore.py --exp_dir $EXP_DIR --scene kitchen1 --model_type LPN_rgb --mapping_every 20
```
This will save the results under `$EXP_DIR/slamcore/kitchen1`

