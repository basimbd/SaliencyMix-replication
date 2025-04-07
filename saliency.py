import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def get_salient_coordinates(image, lamda):
    # print(f"Image shape: {image.size()}")

    # if image.ndim == 4:
    #     _, _, height, width = image.shape
    #     cv2_image = image.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
    # else:
    _, height, width = image.shape
    cv2_image = image.permute(1, 2, 0).cpu().numpy() * 255

    cut_ratio = np.sqrt(1. - lamda)
    # print(f"Cut ratio: {cut_ratio}")
    cut_h, cut_w = int(cut_ratio * height), int(cut_ratio * width)
    # print(f"height, width = {height, width}")
    # print(f"cut (h,w) = {cut_h, cut_w}")

    
    cv2_image = cv2_image[..., ::-1]        # CV2 expects BGR instead of RGB

    saliency_model = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency_model.computeSaliency(cv2_image)
    assert saliency_map.shape == (height, width), "Saliency map shape mismatch"
    # # print(saliency_map.shape)
    # print(f"max index: {np.argmax(saliency_map, axis=None)}")
    y, x = np.unravel_index(np.argmax(saliency_map, axis=None), saliency_map.shape)
    # print(f"x,y = {x, y}")
    x1, x2 = max(0, x - cut_w // 2), min(width, x + cut_w // 2)
    y1, y2 = max(0, y - cut_h // 2), min(height, y + cut_h // 2)
    # print(x1, x2, y1, y2)
    # new_image = image[:, y1:y2, x1:x2]*255
    # print(f"new image shape = {new_image.shape}")
    # transforms.ToPILImage()(new_image.byte()).show()
    return x1, x2, y1, y2

if __name__ == '__main__':
    image = Image.open("cat-close-up-of-side-profile.jpg")
    # print(f"(w, h) = {image.size}")
    image = transforms.ToTensor()(image).unsqueeze(0).cuda()
    # print(f"Image tensor shape = {image.shape}")
    # image = torch.distributions.Beta(1, 2).sample((3, 112, 112))
    get_salient_coordinates(image[0], 0.8)
