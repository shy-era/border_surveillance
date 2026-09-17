import os
import glob
import random
from typing import Tuple, List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SatelliteChangeDataset(Dataset):
    """
    PyTorch Dataset for Bi-temporal Satellite Change Detection.
    Supports standard benchmark layouts (LEVIR-CD, WHU, OSCD):
      root_dir/
        ├── A/ or before/  (Image T1 - Pre-change)
        ├── B/ or after/   (Image T2 - Post-change)
        └── label/ or mask/ (Binary Change Mask)
    """
    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        is_train: bool = True,
        transform: bool = True
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train
        self.transform = transform

        self.dir_a, self.dir_b, self.dir_mask = self._find_subdirectories(root_dir)
        self.image_pairs = self._match_image_pairs()

        if len(self.image_pairs) == 0:
            print(f"Warning: No valid image pairs found in {root_dir}")

    def _find_subdirectories(self, root: str) -> Tuple[str, str, Optional[str]]:
        # Candidate names
        a_candidates = ['A', 'before', 'pre', 't1', 'time1']
        b_candidates = ['B', 'after', 'post', 't2', 'time2']
        mask_candidates = ['label', 'label_binary', 'mask', 'masks', 'labels', 'out', 'ground_truth']

        def _search_in(dir_path: str):
            da, db, dm = None, None, None
            if not os.path.exists(dir_path):
                return None, None, None
            for cand in a_candidates:
                p = os.path.join(dir_path, cand)
                if os.path.isdir(p):
                    da = p
                    break
            for cand in b_candidates:
                p = os.path.join(dir_path, cand)
                if os.path.isdir(p):
                    db = p
                    break
            for cand in mask_candidates:
                p = os.path.join(dir_path, cand)
                if os.path.isdir(p):
                    dm = p
                    break
            return da, db, dm

        # 1. Direct search in provided root
        dir_a, dir_b, dir_mask = _search_in(root)

        # 2. If not found, check subdirectories or common dataset folders
        if not dir_a or not dir_b:
            fallback_dirs = [
                os.path.join('data', 'samples'),
                os.path.join('data', 'LEVIR-CD'),
                'data',
                os.path.join(os.path.dirname(root), 'samples'),
                os.path.join(os.path.dirname(root), 'LEVIR-CD')
            ]
            for fb in fallback_dirs:
                if os.path.isdir(fb):
                    da, db, dm = _search_in(fb)
                    if da and db:
                        dir_a, dir_b, dir_mask = da, db, dm
                        print(f"[Dataset] Note: Redirected data search to detected directory: {fb}")
                        break

        # 3. If still not found, recursive search
        if not dir_a or not dir_b:
            for parent, dirs, _ in os.walk('data' if os.path.exists('data') else '.'):
                da, db, dm = _search_in(parent)
                if da and db:
                    dir_a, dir_b, dir_mask = da, db, dm
                    print(f"[Dataset] Note: Found dataset via scan in: {parent}")
                    break

        return dir_a or root, dir_b or root, dir_mask

    def _match_image_pairs(self) -> List[Tuple[str, str, Optional[str]]]:
        pairs = []
        if not os.path.exists(self.dir_a):
            return pairs

        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        filenames = [
            f for f in os.listdir(self.dir_a) 
            if not f.startswith('.') and os.path.splitext(f)[1].lower() in valid_exts
        ]

        for fname in sorted(filenames):
            path_a = os.path.join(self.dir_a, fname)
            path_b = os.path.join(self.dir_b, fname)
            path_mask = os.path.join(self.dir_mask, fname) if self.dir_mask else None

            if not os.path.exists(path_b):
                # Try matching by base name
                base_name = os.path.splitext(fname)[0]
                matched_b = glob.glob(os.path.join(self.dir_b, f"{base_name}.*"))
                if matched_b:
                    path_b = matched_b[0]
                else:
                    continue

            if path_mask and not os.path.exists(path_mask):
                base_name = os.path.splitext(fname)[0]
                matched_mask = glob.glob(os.path.join(self.dir_mask, f"{base_name}.*"))
                path_mask = matched_mask[0] if matched_mask else None

            pairs.append((path_a, path_b, path_mask))

        return pairs

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
        path_a, path_b, path_mask = self.image_pairs[idx]
        img_a = Image.open(path_a).convert('RGB')
        img_b = Image.open(path_b).convert('RGB')

        mask = None
        if path_mask and os.path.exists(path_mask):
            mask = Image.open(path_mask).convert('L')

        t_a, t_b, t_mask = self._apply_augmentations(img_a, img_b, mask)
        sample_name = os.path.splitext(os.path.basename(path_a))[0]

        return t_a, t_b, t_mask, sample_name
