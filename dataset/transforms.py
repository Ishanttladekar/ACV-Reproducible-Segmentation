import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image

class JointTransform:
    def __init__(self, is_train=True, img_size=(128, 128)):
        self.is_train = is_train
        self.img_size = img_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-10.0, 10.0)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
                image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.int64).squeeze(0)
        mask_semantic = (mask_tensor > 0).long()

        return image, mask_semantic
