import os
import glob
import random
from typing import Tuple, List, Optional, Union
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SatelliteChangeDataset(Dataset):
    """
    PyTorch Dataset for Bi-temporal Satellite Change Detection.
    Supports standard benchmark layouts (LEVIR-CD, S2Looking, WHU, OSCD):
      root_dir/
        ├── A/ or Image1/ or before/  (Image T1 - Pre-change)
        ├── B/ or Image2/ or after/   (Image T2 - Post-change)
        └── label/ or mask/           (Binary Change Mask)
    Can also accept a list of directories or automatically aggregate all dataset folders under `data/`.
    """
    def __init__(
        self,
        root_dir: Union[str, List[str]] = 'data',
        image_size: int = 512,
        is_train: bool = True,
        transform: bool = True
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.transform = transform
        self.image_pairs: List[Tuple[str, str, Optional[str]]] = []

        # Candidate folder names across all datasets (LEVIR-CD, S2Looking, WHU)
        self.a_candidates = ['A', 'Image1', 'image1', 'img1', 'before', 'pre', 't1', 'time1']
        self.b_candidates = ['B', 'Image2', 'image2', 'img2', 'after', 'post', 't2', 'time2']
        self.mask_candidates = ['label', 'label_binary', 'mask', 'masks', 'labels', 'out', 'ground_truth']

        dirs_to_scan = [root_dir] if isinstance(root_dir, str) else root_dir
        
        for d in dirs_to_scan:
            self._discover_and_add_pairs(d)

        print(f"[Dataset] Total combined bi-temporal pairs loaded: {len(self.image_pairs)}")

    def _discover_and_add_pairs(self, root: str):
        if not os.path.exists(root):
            return

        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

        # Helper to check if a directory has standard A/B/label structure
        def _check_dir(da, db, dm):
            if not da or not db or not os.path.isdir(da) or not os.path.isdir(db):
                return
            filenames = [
                f for f in os.listdir(da)
                if not f.startswith('.') and os.path.splitext(f)[1].lower() in valid_exts
            ]
            added = 0
            for fname in filenames:
                path_a = os.path.join(da, fname)
                path_b = os.path.join(db, fname)
                path_mask = os.path.join(dm, fname) if (dm and os.path.isdir(dm)) else None

                if not os.path.exists(path_b):
                    base_name = os.path.splitext(fname)[0]
                    matched_b = glob.glob(os.path.join(db, f"{base_name}.*"))
                    if matched_b:
                        path_b = matched_b[0]
                    else:
                        continue

                if path_mask and not os.path.exists(path_mask):
                    base_name = os.path.splitext(fname)[0]
                    matched_mask = glob.glob(os.path.join(dm, f"{base_name}.*"))
                    path_mask = matched_mask[0] if matched_mask else None

                self.image_pairs.append((path_a, path_b, path_mask, None))
                added += 1
            if added > 0:
                print(f"[Dataset] Loaded {added} pairs from: {da} & {db}")

        # Check for WHU Building Change Detection Dataset layout
        for split in ['train', 'test']:
            whu_patterns = [
                os.path.join(root, '**', '1. The two-period image data'),
                os.path.join(root, '1. The two-period image data'),
                os.path.join(root, '**', 'WHU*')
            ]
            for pat in whu_patterns:
                for match_dir in glob.glob(pat, recursive=True):
                    whu_a_img = os.path.join(match_dir, '2012', 'splited_images', split, 'image')
                    whu_b_img = os.path.join(match_dir, '2016', 'splited_images', split, 'image')
                    whu_a_lbl = os.path.join(match_dir, '2012', 'splited_images', split, 'label')
                    whu_b_lbl = os.path.join(match_dir, '2016', 'splited_images', split, 'label')

                    if os.path.isdir(whu_a_img) and os.path.isdir(whu_b_img):
                        filenames = [f for f in os.listdir(whu_a_img) if os.path.splitext(f)[1].lower() in valid_exts]
                        whu_added = 0
                        for fname in filenames:
                            pa = os.path.join(whu_a_img, fname)
                            pb = os.path.join(whu_b_img, fname)
                            pla = os.path.join(whu_a_lbl, fname) if os.path.isdir(whu_a_lbl) else None
                            plb = os.path.join(whu_b_lbl, fname) if os.path.isdir(whu_b_lbl) else None
                            if os.path.exists(pb):
                                self.image_pairs.append((pa, pb, pla, plb))
                                whu_added += 1
                        if whu_added > 0:
                            print(f"[Dataset] Loaded {whu_added} WHU [{split}] pairs from {match_dir}")

        # Standard check for LEVIR-CD and S2Looking directories
        known_roots = [
            (os.path.join(root, 'samples', 'A'), os.path.join(root, 'samples', 'B'), os.path.join(root, 'samples', 'label')),
            (os.path.join(root, 'A'), os.path.join(root, 'B'), os.path.join(root, 'label')),
            (os.path.join(root, 'samples', 'S2Looking', 'Image1'), os.path.join(root, 'samples', 'S2Looking', 'Image2'), os.path.join(root, 'samples', 'S2Looking', 'label')),
            (os.path.join(root, 'S2Looking', 'Image1'), os.path.join(root, 'S2Looking', 'Image2'), os.path.join(root, 'S2Looking', 'label')),
            (os.path.join(root, 's2looking', 'Image1'), os.path.join(root, 's2looking', 'Image2'), os.path.join(root, 's2looking', 'label'))
        ]
        for da, db, dm in known_roots:
            _check_dir(da, db, dm)

    def __len__(self) -> int:
        return len(self.image_pairs)

    def _apply_augmentations(self, img_a: Image.Image, img_b: Image.Image, mask: Optional[Image.Image]):
        # Resize first
        img_a = img_a.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_b = img_b.resize((self.image_size, self.image_size), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.is_train and self.transform:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_a = TF.hflip(img_a)
                img_b = TF.hflip(img_b)
                if mask is not None:
                    mask = TF.hflip(mask)

            # Random Vertical Flip
            if random.random() > 0.5:
                img_a = TF.vflip(img_a)
                img_b = TF.vflip(img_b)
                if mask is not None:
                    mask = TF.vflip(mask)

            # Random 90 degree rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img_a = TF.rotate(img_a, angle)
                img_b = TF.rotate(img_b, angle)
                if mask is not None:
                    mask = TF.rotate(mask, angle)

        # Convert to Tensor & Normalize
        t_a = TF.to_tensor(img_a)
        t_b = TF.to_tensor(img_b)

        # Ensure 3 channels
        if t_a.shape[0] == 1:
            t_a = t_a.repeat(3, 1, 1)
        if t_b.shape[0] == 1:
            t_b = t_b.repeat(3, 1, 1)

        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t_a = TF.normalize(t_a, mean=mean, std=std)
        t_b = TF.normalize(t_b, mean=mean, std=std)

        if mask is not None:
            mask_arr = np.array(mask)
            # Threshold to binary (0 or 1)
            t_mask = torch.from_numpy((mask_arr > 127).astype(np.float32)).unsqueeze(0)
        else:
            t_mask = torch.zeros(1, self.image_size, self.image_size)

        return t_a, t_b, t_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item = self.image_pairs[idx]
        if len(item) == 4:
            path_a, path_b, path_mask_a, path_mask_b = item
        else:
            path_a, path_b, path_mask_a = item
            path_mask_b = None

        img_a = Image.open(path_a).convert('RGB')
        img_b = Image.open(path_b).convert('RGB')

        mask = None
        if path_mask_b and os.path.exists(path_mask_b) and path_mask_a and os.path.exists(path_mask_a):
            # Compute change mask dynamically between two temporal building masks (WHU)
            ma = np.array(Image.open(path_mask_a).convert('L')) > 128
            mb = np.array(Image.open(path_mask_b).convert('L')) > 128
            change_arr = (np.bitwise_xor(ma, mb).astype(np.uint8)) * 255
            mask = Image.fromarray(change_arr)
        elif path_mask_a and os.path.exists(path_mask_a):
            mask = Image.open(path_mask_a).convert('L')

        # Negative Pair Augmentation during training (25% probability):
        # Pass identical image with lighting/color jitter and zero-change mask
        # Teaches Siamese network to be strictly invariant to lighting & natural unchanged terrain
        if self.is_train and random.random() < 0.25:
            # Pick img_a or img_b as reference
            base_img = img_a if random.random() > 0.5 else img_b
            # Create perturbed version with slight brightness/contrast/hue jitter
            jittered = base_img.copy()
            if random.random() > 0.5:
                jittered = TF.adjust_brightness(jittered, random.uniform(0.8, 1.25))
            if random.random() > 0.5:
                jittered = TF.adjust_contrast(jittered, random.uniform(0.85, 1.2))
            if random.random() > 0.5:
                jittered = TF.adjust_saturation(jittered, random.uniform(0.8, 1.2))

            t_a, t_b, _ = self._apply_augmentations(base_img, jittered, None)
            t_mask = torch.zeros(1, self.image_size, self.image_size)
            sample_name = f"neg_{os.path.splitext(os.path.basename(path_a))[0]}"
            return t_a, t_b, t_mask, sample_name

        t_a, t_b, t_mask = self._apply_augmentations(img_a, img_b, mask)
        sample_name = os.path.splitext(os.path.basename(path_a))[0]

        return t_a, t_b, t_mask, sample_name
