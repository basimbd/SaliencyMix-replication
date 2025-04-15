import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets import VOCDetection
import torchvision.transforms
import detection.transforms as T

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class VOCDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, voc_dataset, transforms):
        self.voc_dataset = voc_dataset
        self.transforms = transforms
        self.class_to_idx = {cls_name: idx + 1 for idx, cls_name in enumerate(VOC_CLASSES)}

    def __getitem__(self, idx):
        img, target = self.voc_dataset[idx]
        target = self._convert_voc_target(target, idx)

        img = torchvision.transforms.functional.to_tensor(img)

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.voc_dataset)

    def _convert_voc_target(self, target, idx):
        boxes = []
        labels = []

        objects = target["annotation"]["object"]
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])

            label_str = obj["name"].lower().strip()
            labels.append(self.class_to_idx[label_str])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        converted = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return converted

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_voc_datasets(test_only=False):
    voc_test_raw = VOCDetection('VOCdevkit', year='2007', image_set='test', download=True)
    transform_test = get_transform(train=False)
    dataset_test = VOCDatasetWrapper(voc_test_raw, transform_test)
    
    if test_only:
        return dataset_test
    voc_2007_raw = VOCDetection('VOCdevkit', year='2007', image_set='trainval', download=True)
    voc_2012_raw = VOCDetection('VOCdevkit', year='2012', image_set='trainval', download=True)

    transform_train = get_transform(train=True)
    dataset_2007 = VOCDatasetWrapper(voc_2007_raw, transform_train)
    dataset_2012 = VOCDatasetWrapper(voc_2012_raw, transform_train)
    dataset = ConcatDataset([dataset_2007, dataset_2012])
 
    return dataset, dataset_test
 