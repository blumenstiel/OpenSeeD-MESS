# Multi-domain Evaluation of Semantic Segmentation (MESS) with SAN

[[Website](https://github.io)] [[arXiv](https://arxiv.org/)] [[GitHub](https://github.com/blumenstiel/MESS)]

This directory contains the code for the MESS evaluation of SAN. Please see the commits for our changes of the model.

## Setup
Create a conda environment `san` and install the required packages. See [mess/README.md]([mess/README.md]) for details.
```sh
 bash mess/setup_env.sh
```

Prepare the datasets by following the instructions in [mess/DATASETS.md](mess/DATASETS.md). The `san` env can be used for the dataset preparation. If you evaluate multiple models with MESS, you can change the `dataset_dir` argument and the `DETECTRON2_DATASETS` environment variable to a common directory (see [mess/DATASETS.md](mess/DATASETS.md) and [mess/eval.sh](mess/eval.sh)). 

Download the SAN weights with
```sh
mkdir weights
wget https://huggingface.co/Mendel192/san/resolve/main/san_vit_b_16.pth -O weights/san_vit_b_16.pth
wget https://huggingface.co/Mendel192/san/resolve/main/san_vit_large_14.pth -O weights/san_vit_large_14.pth
```

## Evaluation
To evaluate the SAN models on the MESS dataset, run
```sh
bash mess/eval.sh

# for evaluation in the background:
nohup bash mess/eval.sh > eval.log &
tail -f eval.log 
```

For evaluating a single dataset, select the DATASET from [mess/DATASETS.md](mess/DATASETS.md), the DETECTRON2_DATASETS path, and run
```
conda activate san
export DETECTRON2_DATASETS="datasets"
DATASET=<dataset_name>

# Base model
python train_net.py --eval-only --num-gpus 1 --config-file configs/san_clip_vit_res4_coco.yaml OUTPUT_DIR output/SAN_base/$DATASET MODEL.WEIGHTS weights/san_vit_b_16.pth DATASETS.TEST \(\"$DATASET\",\)
# Large model
python train_net.py --eval-only --num-gpus 1 --config-file configs/san_clip_vit_large_res4_coco.yaml OUTPUT_DIR output/SAN_large/$DATASET MODEL.WEIGHTS weights/san_vit_large_14.pth DATASETS.TEST \(\"$DATASET\",\)
```

# --- Original SAN README.md ---

# OpenSeeD
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-framework-for-open-vocabulary/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=a-simple-framework-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-framework-for-open-vocabulary/panoptic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-ade20k-val?p=a-simple-framework-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-framework-for-open-vocabulary/instance-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/instance-segmentation-on-ade20k-val?p=a-simple-framework-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-framework-for-open-vocabulary/instance-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/instance-segmentation-on-cityscapes-val?p=a-simple-framework-for-open-vocabulary)

This is the official implementation of the paper "[A Simple Framework for Open-Vocabulary Segmentation and Detection](https://arxiv.org/pdf/2303.08131.pdf)".

https://user-images.githubusercontent.com/34880758/225408795-d1e714e0-cfc8-4466-b052-045d54409a1d.mp4

You can also find the more detailed demo at [video link on Youtube](https://www.youtube.com/watch?v=z4gsQw2n7iM).

:point_right: **[New] demo code is available**

### :rocket: Key Features
- A Simple Framework for Open-Vocabulary Segmentation and Detection.
- Support interactive segmentation with box input to generate mask.

### :bulb: Installation
```sh
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install -r requirements.txt
sh install_cococapeval.sh
export DATASET=/pth/to/dataset
```
Download the pretrained checkpoint from [here](https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt).
### :bulb: Demo script
```sh
python demo/demo_panoseg.py evaluate --conf_files configs/openseed/openseed_swint_lang.yaml  --image_path images/your_image.jpg --overrides WEIGHT /path/to/ckpt/model_state_dict_swint_51.2ap.pt
```
:fire: Remember to **modify the vocabulary**  `thing_classes` and `stuff_classes` in `demo_panoseg.py`  if your want to segment open-vocabulary objects.

**Evaluation on coco**
```sh
mpirun -n 1 python eval_openseed.py evaluate --conf_files configs/openseed/openseed_swint_lang.yaml  --overrides WEIGHT /path/to/ckpt/model_state_dict_swint_51.2ap.pt COCO.TEST.BATCH_SIZE_TOTAL 2
```
You are expected to get `55.4` PQ.

![hero_figure](figs/intro.jpg)
### :unicorn: Model Framework
![hero_figure](figs/framework.jpg)
### :volcano: Results
Results on open segmentation
![hero_figure](figs/results1.jpg)
Results on task transfer and segmentation in the wild
![hero_figure](figs/results2.jpg)


### <a name="CitingOpenSeeD"></a>Citing OpenSeeD

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@article{zhang2023simple,
  title={A Simple Framework for Open-Vocabulary Segmentation and Detection},
  author={Zhang, Hao and Li, Feng and Zou, Xueyan and Liu, Shilong and Li, Chunyuan and Gao, Jianfeng and Yang, Jianwei and Zhang, Lei},
  journal={arXiv preprint arXiv:2303.08131},
  year={2023}
}
```

