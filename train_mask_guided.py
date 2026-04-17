import os
import math
import random
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ------------------------
# Synthetic dataset with subject-darkening
# ------------------------

class MaskGuidedLowLightDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, size: Tuple[int, int]=(256, 256)):
        self.root_dir = root_dir
        self.size = size
        self.transform = transform
        self.images = []
        exts = {'.jpg','.jpeg','.png','.bmp'}
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in exts:
                    self.images.append(os.path.join(dirpath, fn))

    def __len__(self):
        return len(self.images)

    def _random_subject_mask(self, h: int, w: int) -> np.ndarray:
        # Random blob using Gaussian kernels + threshold
        yy, xx = np.mgrid[0:h, 0:w]
        mask = np.zeros((h, w), dtype=np.float32)
        num_blobs = random.randint(1, 3)
        for _ in range(num_blobs):
            cy = random.uniform(0.3*h, 0.7*h)
            cx = random.uniform(0.3*w, 0.7*w)
            sy = random.uniform(0.15*h, 0.35*h)
            sx = random.uniform(0.15*w, 0.35*w)
            blob = np.exp(-(((yy-cy)**2)/(2*sy*sy) + ((xx-cx)**2)/(2*sx*sx)))
            mask = np.maximum(mask, blob)
        # Smooth and threshold
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        th = random.uniform(0.3, 0.6)
        mask = (mask > th).astype(np.float32)
        # Edge-smooth
        import cv2
        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3)
        mask = np.clip(mask, 0, 1)
        return mask

    def _apply_darkening(self, img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Darken subject only: gamma > 1 on masked area + reduce exposure
        dark_gamma = random.uniform(1.6, 2.2)
        exposure_scale = random.uniform(0.5, 0.8)
        img = img_np.astype(np.float32) / 255.0
        masked = img ** dark_gamma
        masked = masked * exposure_scale
        mask3 = mask[..., None]
        out = img * (1 - mask3) + masked * mask3
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        img = Image.open(path).convert('RGB')
        if self.size:
            img = img.resize(self.size, Image.BICUBIC)
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        mask = self._random_subject_mask(h, w)
        low_np = self._apply_darkening(img_np, mask)

        gt = Image.fromarray(img_np)
        low = Image.fromarray(low_np)
        mask_img = Image.fromarray((mask*255).astype(np.uint8))

        if self.transform is None:
            to_tensor = transforms.ToTensor()
            gt_t = to_tensor(gt)
            low_t = to_tensor(low)
            mask_t = to_tensor(mask_img)
        else:
            gt_t = self.transform(gt)
            low_t = self.transform(low)
            mask_t = transforms.ToTensor()(mask_img)

        return {'low': low_t, 'gt': gt_t, 'mask': mask_t}


# ------------------------
# Simple DEDUNet head with mask prediction and gated enhancement
# ------------------------

class SimpleUNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        c = base
        self.enc1 = nn.Sequential(nn.Conv2d(3, c, 3, 1, 1), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(c, 2*c, 3, 1, 1), nn.ReLU(), nn.Conv2d(2*c, 2*c, 3, 1, 1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bott = nn.Sequential(nn.Conv2d(2*c, 4*c, 3, 1, 1), nn.ReLU(), nn.Conv2d(4*c, 4*c, 3, 1, 1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(4*c, 2*c, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(4*c, 2*c, 3, 1, 1), nn.ReLU(), nn.Conv2d(2*c, 2*c, 3, 1, 1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(2*c, c, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(2*c, c, 3, 1, 1), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1), nn.ReLU())
        self.out_img = nn.Conv2d(c, 3, 1)
        self.out_mask = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bott(p2)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        img = torch.tanh(self.out_img(d1))
        mask = torch.sigmoid(self.out_mask(d1))
        return img, mask


class GatedEnhancer(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.unet = SimpleUNet(base)

    def forward(self, x):
        enh, m = self.unet(x)
        # Gate: out = x + m * (enh - x)
        out = x + m * (enh - x)
        return out, m


# ------------------------
# Losses
# ------------------------

class MaskGuidedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred_img, pred_mask, gt, inp):
        # Subject loss: encourage improvement where mask is high
        # Approx subject supervision: gt vs pred while preserving background
        with torch.no_grad():
            # Estimate brightness as proxy for background (preserve where input bright)
            bright_bg = (inp.mean(dim=1, keepdim=True) > 0.6).float()
        subject_w = pred_mask.detach()
        background_w = bright_bg

        l_subject = self.l1(pred_img * subject_w, gt * subject_w)
        l_background_preserve = self.l1(pred_img * background_w, inp * background_w)
        l_mask_sparsity = pred_mask.mean()
        loss = l_subject + 0.5 * l_background_preserve + 0.05 * l_mask_sparsity
        return loss, {
            'l_subject': l_subject.detach(),
            'l_background': l_background_preserve.detach(),
            'l_mask': l_mask_sparsity.detach()
        }


# ------------------------
# Trainer
# ------------------------

def train(root_dir: str, epochs: int = 5, batch_size: int = 4, lr: float = 2e-4, out_dir: str = 'checkpoints_mask_guided'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = MaskGuidedLowLightDataset(root_dir)
    if len(ds) == 0:
        raise ValueError(f"No images found in '{root_dir}'. Provide a folder containing images (jpg/png/bmp). Example: python train_mask_guided.py --data C:/Users/Akash/Pictures --epochs 10")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = GatedEnhancer(base=32).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = MaskGuidedLoss()

    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for i, batch in enumerate(dl, 1):
            low = batch['low'].to(device)
            gt = batch['gt'].to(device)
            out, m = model(low)
            loss, parts = criterion(out, m, gt, low)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch} Iter {i}: loss {running / i:.4f} subj {parts['l_subject']:.4f} bg {parts['l_background']:.4f} mask {parts['l_mask']:.4f}")
        torch.save(model.state_dict(), os.path.join(out_dir, f"mask_guided_e{epoch}.pth"))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Path to folder of normal images for synthetic darkening')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--out', type=str, default='checkpoints_mask_guided')
    args = ap.parse_args()
    train(args.data, args.epochs, args.batch_size, args.lr, args.out)


