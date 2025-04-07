import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def get_cutmix_coordinates(image, lamda):
    _, height, width = image.shape
    cut_ratio = np.sqrt(1. - lamda)
    cut_h, cut_w = int(cut_ratio * height), int(cut_ratio * width)

    x = torch.randint(0, width, (1,)).item()
    y = torch.randint(0, height, (1,)).item()

    x1, x2 = max(0, x - cut_w // 2), min(width, x + cut_w // 2)
    y1, y2 = max(0, y - cut_h // 2), min(height, y + cut_h // 2)
    return x1, x2, y1, y2

if __name__ == '__main__':
    image = Image.open("cat-close-up-of-side-profile.jpg")
    # print(f"(w, h) = {image.size}")
    image = transforms.ToTensor()(image).unsqueeze(0).cuda()
    get_cutmix_coordinates(image[0], 0.8)
    