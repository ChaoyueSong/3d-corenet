# 3D Pose Transfer with Correspondence Learning and Mesh Refinement

#### [Project](https://chaoyuesong.github.io/3d-corenet/) |   [Paper](https://openreview.net/pdf?id=fG01Z_unHC) |  [Video](https://www.youtube.com/watch?v=HjBvPhZSl-o) | [Video (in Chinese)](https://www.bilibili.com/video/BV1rq4y1q7pZ?spm_id_from=333.999.0.0)

**3D Pose Transfer with Correspondence Learning and Mesh Refinement** <br>
Chaoyue Song,
Jiacheng Wei,
Ruibo Li,
[Fayao Liu](https://sites.google.com/site/fayaoliu/),
[Guosheng Lin](https://guosheng.github.io/) <br>
in NeurIPS, 2021.

<img src="files/3dpt.gif" width="60%" height="60%" /> <br>


## News
* `21/11/2022` We release our latest work on unsupervised 3D pose transfer. Please check it [here](https://arxiv.org/abs/2211.10278). In this paper, we present X-DualNet, an unsupervised deep learning framework to solve the 3D pose transfer problem in an end-to-end fashion. Through extensive experiments on human and animal meshes, we demonstrate X-DualNet achieves comparable performance as the state-of-the-art supervised approaches qualitatively and quantitatively and even outperforms some of them. The code of X-DualNet will be released soon at [here](https://github.com/ChaoyueSong/X-DualNet).

## Installation
- Clone this repo:
```bash
git clone https://github.com/ChaoyueSong/3d-corenet.git
cd 3d-corenet
```

- Install the dependencies. Our code has been tested on Python 3.6, PyTorch 1.8 (previous versions also work). And we also need pymesh.
```bash
conda install -c conda-forge pymesh2
```

- Clone the Synchronized-BatchNorm-PyTorch repo.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## Dataset preparation
We use [SMPL](https://smpl.is.tue.mpg.de/) as the human mesh data, please download data [here](https://drive.google.com/drive/folders/11LbPXbDg4F_pSIr0sMHzWI08FOC8XvSY). And we generate our animal mesh data using [SMAL](https://smal.is.tue.mpg.de/), please download it [here](https://drive.google.com/drive/folders/1uP6H0j7mUJ6utgvXxpT-2rn4EYhJ3el5?usp=sharing).

## Generating Meshes Using Pretrained model
By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

#### 1) SMPL (human) 
Download the pretrained model from [pretrained model link](https://drive.google.com/drive/folders/1pZqw_AU7VpVOnop6HSv6WeRGfCEPt2lm?usp=sharing) and save them in `checkpoints/human`. Then run the command  
````bash
python test.py --dataset_mode human --dataroot [Your data path] --gpu_ids 0
````
The results will be saved in `test_results/human/` by default. `human_test_list` is randomly choosed for testing.

#### 2) SMAL (animal) 
Download the pretrained model from [pretrained model link](https://drive.google.com/drive/folders/1v3Iz51MtcYsLoKAu9XAdFG6JmhxLHWJ-?usp=sharing) and save them in `checkpoints/animal`. Then run the command 
````bash
python test.py --dataset_mode animal --dataroot [Your data path] --gpu_ids 0
````
The results will be saved in `test_results/animal/` by default. `animal_test_list` is randomly choosed for testing.

## Training
#### 1) SMPL (human) 
To train new models on human meshes, please run:
```bash
python train.py --dataset_mode human --dataroot [Your data path] --niter 100 --niter_decay 100 --batchSize 8 --gpu_ids 0,1
```
The output meshes in the training process will be saved in `output/human/`.
#### 2) SMAL (animal) 
To train new models on animal meshes, please run:
```bash
python train.py --dataset_mode animal --dataroot [Your data path] --niter 100 --niter_decay 100 --batchSize 8 --gpu_ids 0,1
```
The output meshes in the training process will be saved in `output/animal/`.


Please change the batch size and gpu_ids as you desired.

If you need continue training from checkpoint, use `--continue_train`.
## Citation
If you use this code for your research, please cite the following work.

```bash
@inproceedings{song20213d,
  title={3D Pose Transfer with Correspondence Learning and Mesh Refinement},
  author={Song, Chaoyue and Wei, Jiacheng and Li, Ruibo and Liu, Fayao and Lin, Guosheng},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

## Acknowledgments

This code is heavily based on [CoCosNet](https://github.com/microsoft/CoCosNet). We rewrite the [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) architecture to ver2ver. We also use Optimal Transport code from [FLOT](https://github.com/valeoai/FLOT), Data and Edge loss code from [NPT](https://github.com/jiashunwang/Neural-Pose-Transfer), and [Synchronized Batch Normalization](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).

We thank all authors for the wonderful code!



