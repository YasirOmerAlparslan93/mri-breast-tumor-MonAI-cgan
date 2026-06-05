import os
import random
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader

THRESHOLD = 0.5

def list_image_files(root_path):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    for root, dirs, filenames in os.walk(root_path):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return sorted(files)

class BTSDataset(Dataset):
    def __init__(self, pairs, img_size=512, augment=False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = TF.resize(img, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size))

        # FIXED SYNCHRONIZED AUGMENTATION
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        # COLOR JITTER ONLY IMAGE
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        mask = TF.to_tensor(mask)
        mask = (mask > THRESHOLD).float()

        return img, mask

def get_dataloaders(image_path, mask_path, img_size=512, batch_size=8, num_workers=8):
    image_files = list_image_files(image_path)
    mask_files = list_image_files(mask_path)

    mask_map = {os.path.splitext(os.path.basename(m))[0]: m for m in mask_files}
    pairs = [(img, mask_map[os.path.splitext(os.path.basename(img))[0]]) for img in image_files if os.path.splitext(os.path.basename(img))[0] in mask_map]

    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]

    train_loader = DataLoader(BTSDataset(train_pairs, img_size, augment=True), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(BTSDataset(val_pairs, img_size, augment=False), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(BTSDataset(test_pairs, img_size, augment=False), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader