# SaliencyMix-replication
This repository is a replication of the experiments from the SaliencyMix [paper](https://openreview.net/pdf?id=-M0QkvBGTTq) from ICLR 2021. This paper proposes a SaliencyMix data augmentation technique for improved regularization. It is similar to other mixing techniques, but instead of randomly selecting patch, it selects the most salient region of the source image as patch, and as a result retains more useful information.

In this repo, we reproduce the experiments in this paper. The experiments can be divided in 4 parts. CIFAR classification, ImageNet classification, Object Detection, and Class Activation Map generation. Below we introduce each of them briefly.

## Installation
But before you begin, you may want to install the packages in the `requirements.txt` file with the following command-
```
pip install -r requirements.txt
```

## CIFAR10 and CIFAR100
For the CIFAR dataset, we create a basic training script `train.py` and ResNet50, ResNet101, WideResNet-28-10 models in `models/resnet.py` and `models/wideresnet.py` files. Saliency map generation for saliency mixing can be found in `saliency.py`. Similarly other baseline augmentations are implemented in `cutmix.py` and `cutout.py` files. To start training any of these models on CIFAR data, the following command can be useful as a template-
```
python train.py --model resnet50 --dataset cifar100 --saliency_probab 0.5 --batch_size 128 --epochs 200
```
If you want to resume training from a previous checkpoint, add `--checkpoint <PATH>` to the command. To evaluate an already trained model on CIFAR, the following command can be helpful-
```
python evaluate.py --model resnet50 --dataset cifar100 --model_path <PATH>
```

## ImageNet
For the ImageNet dataset, the data must already be downloaded. After downloading the data, keep the train and evaluation set under `imagenet/train` and `imagenet/val` directory (replace `imagenet/` with `tiny-imagenet/` if `tiny-imagenet` dataset is to be used). Then, similar to the previous training, run a command similar to the following-
```
python imgnet_train.py --model resnet50 --dataset imagenet --saliency_probab 0.5 --batch_size 256 --epochs 300
```
The evaluation command can be same as before.

Parts of CIFAR and ImageNet implementations are extended/inspired from the official [CutMix](https://github.com/clovaai/CutMix-PyTorch) and [CutOut](https://github.com/uoguelph-mlrg/Cutout) repository.

## Object Detection
For object detection, we initially used [this repo](https://github.com/trzy/FasterRCNN) since it is easy to use our custom ResNet models. Later, we used the Faster-RCNN model from PyTorch following this popular [pytorch tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) for our detection. In the process, we take several script in our `detection/` directory from [torchvision/references/detection](https://github.com/pytorch/vision/tree/main/references/detection) as is recommended by the official PyTorch tutorial.

To start fine-tuning the Faster-RCNN model with a pre-trained ResNet backbone, the following command is necessary-
```
python detect.py --backbone-checkpoint <PATH_TO_PRE-TRAINED_MODEL>
```
Other parameters will be set to default or can be passed as CLI args. For evaluation, one needs to add the `--evaluate` option to the above command. Note that our detection script currently only works with PASCAL VOC dataset.

## CAM Generation
For class activation map (CAM) generation from a pre-trained ResNet model, the following command should be used-
```
python cam.py --image_path <PATH_TO_IMAGE> --model_path <PATH_TO_PRE-TRAINED_MODEL>
```
`--model_path` is optional, and pre-trained ResNet50 from torchvision will be used if not provided.
