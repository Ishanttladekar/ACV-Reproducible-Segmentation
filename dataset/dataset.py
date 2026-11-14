import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataset.transforms import JointTransform

class PennFudanPedDataset(Dataset):
    CLASSES = {0: "Background", 1: "Pedestrian"}
    NUM_CLASSES = 2

    def __init__(self, root_dir: str, split: str = "train", seed: int = 42, img_size=(128, 128)):
        self.root_dir = root_dir
        self.split = split.lower()
        self.img_size = img_size

        self.img_dir = os.path.join(root_dir, "PNGImages")
        self.mask_dir = os.path.join(root_dir, "PedMasks")

        all_imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".png")])
        all_masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".png")])
        assert len(all_imgs) == len(all_masks)

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(all_imgs), generator=g).tolist()

        train_indices = perm[:120]
        val_indices = perm[120:145]
        test_indices = perm[145:170]

        if self.split == "train":
            self.indices = train_indices
        elif self.split in ["val", "validation"]:
            self.indices = val_indices
        elif self.split == "test":
            self.indices = test_indices
        else:
            raise ValueError(f"Unknown split: {split}")

        self.image_files = [all_imgs[i] for i in self.indices]
        self.mask_files = [all_masks[i] for i in self.indices]

        is_train = (self.split == "train")
        self.transform = JointTransform(is_train=is_train, img_size=self.img_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image_tensor, mask_tensor = self.transform(image, mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": self.image_files[idx]
        }

def get_dataloaders(data_dir: str, batch_size: int = 8, num_workers: int = 0, seed: int = 42, img_size=(128, 128)):
    train_ds = PennFudanPedDataset(data_dir, split="train", seed=seed, img_size=img_size)
    val_ds = PennFudanPedDataset(data_dir, split="val", seed=seed, img_size=img_size)
    test_ds = PennFudanPedDataset(data_dir, split="test", seed=seed, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
