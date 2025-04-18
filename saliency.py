import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def get_salient_coordinates(image, lamda):
    _, height, width = image.shape
    cut_ratio = np.sqrt(1. - lamda)
    cut_h, cut_w = int(cut_ratio * height), int(cut_ratio * width)
    
    cv2_image = image.permute(1, 2, 0).cpu().numpy() * 255
    cv2_image = cv2_image[..., ::-1]        # CV2 expects BGR instead of RGB

    saliency_model = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency_model.computeSaliency(cv2_image)
    assert saliency_map.shape == (height, width), "Saliency map shape mismatch"
    y, x = np.unravel_index(np.argmax(saliency_map, axis=None), saliency_map.shape)
    x1, x2 = max(0, x - cut_w // 2), min(width, x + cut_w // 2)
    y1, y2 = max(0, y - cut_h // 2), min(height, y + cut_h // 2)
    # new_image = image[:, y1:y2, x1:x2]*255
    # transforms.ToPILImage()(new_image.byte()).show()
    return x1, x2, y1, y2

if __name__ == '__main__':
    image = Image.open("cat-close-up-of-side-profile.jpg")
    # print(f"(w, h) = {image.size}")
    image = transforms.ToTensor()(image).unsqueeze(0).cuda()
    get_salient_coordinates(image[0], 0.8)
