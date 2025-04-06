import torch

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
        print(f"Cutout initialized with n_holes={n_holes}, length={length}")

    def __call__(self, image):
        height = image.size(1)
        width = image.size(2)

        mask = torch.ones(height, width, dtype=torch.float32)

        for n in range(self.n_holes):
            x = torch.randint(0, width, (1,)).item()
            y = torch.randint(0, height, (1,)).item()
            
            x1, x2 = max(0, x - self.length // 2), min(width, x + self.length // 2)
            y1, y2 = max(0, y - self.length // 2), min(height, y + self.length // 2)

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(image)
        image = image * mask

        return image