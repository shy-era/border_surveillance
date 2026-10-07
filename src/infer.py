import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union, Tuple
import cv2

from src.model import SiameseResNetUNet, create_model
from src.postprocess import BorderSurveillancePostProcessor
from src.features import SatelliteFeatureEngineer


class SurveillanceInferenceEngine:
    """
    End-to-End Inference Engine for Border Satellite Change Detection.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        backbone: str = 'resnet18',
        device: Optional[str] = None,
        threshold: float = 0.5,
        min_cluster_area: int = 15,
        gsd_meters_per_pixel: float = 0.5
    ):
        self.threshold = threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.postprocessor = BorderSurveillancePostProcessor(
            min_cluster_area=min_cluster_area,
            gsd_meters_per_pixel=gsd_meters_per_pixel
        )
        self.feature_engineer = SatelliteFeatureEngineer()

        self.model = create_model(backbone=backbone, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.has_trained_weights = False
        if model_path and os.path.isfile(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                # Auto-detect backbone architecture from state dict
                detected_backbone = 'resnet34' if any('layer1.2' in k for k in state_dict.keys()) else 'resnet18'
                if detected_backbone != backbone:
                    print(f"[InferenceEngine] Auto-detected backbone '{detected_backbone}' from checkpoint")
                    self.model = create_model(backbone=detected_backbone, pretrained=False).to(self.device)

                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.has_trained_weights = True
                print(f"[InferenceEngine] Successfully loaded model weights ({detected_backbone}) from {model_path}")
            except Exception as e:
                print(f"[InferenceEngine] Warning: Could not load weights from {model_path}: {e}")

    def preprocess(self, img_a: Image.Image, img_b: Image.Image, target_size: int = 512) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        orig_w, orig_h = img_a.size

        # Resize to model dimensions
        t1_resized = img_a.resize((target_size, target_size), Image.BILINEAR)
        t2_resized = img_b.resize((target_size, target_size), Image.BILINEAR)

        # Convert to Tensor & Normalize
        t1 = TF.to_tensor(t1_resized)
        t2 = TF.to_tensor(t2_resized)

        if t1.shape[0] == 1:
            t1 = t1.repeat(3, 1, 1)
        if t2.shape[0] == 1:
            t2 = t2.repeat(3, 1, 1)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t1 = TF.normalize(t1, mean=mean, std=std).unsqueeze(0).to(self.device)
        t2 = TF.normalize(t2, mean=mean, std=std).unsqueeze(0).to(self.device)

        return t1, t2, (orig_w, orig_h)

    def _computer_vision_baseline(self, img_a: Image.Image, img_b: Image.Image) -> np.ndarray:
        """
        High-precision classical CV difference detector used as an accurate fallback
        or baseline comparison when weights are still training.
        """
        a_np = np.array(img_a.convert('RGB'))
        b_np = np.array(img_b.convert('RGB'))

        gray_a = cv2.cvtColor(a_np, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(b_np, cv2.COLOR_RGB2GRAY)

        # Gaussian blur to reduce high frequency satellite grain
        blur_a = cv2.GaussianBlur(gray_a, (5, 5), 0)
        blur_b = cv2.GaussianBlur(gray_b, (5, 5), 0)

        # Structural gradient difference
        diff = cv2.absdiff(blur_a, blur_b)
        sobel_a = cv2.Sobel(blur_a, cv2.CV_64F, 1, 1, ksize=3)
        sobel_b = cv2.Sobel(blur_b, cv2.CV_64F, 1, 1, ksize=3)
        edge_diff = np.abs(sobel_a - sobel_b)
        edge_diff = np.clip(edge_diff, 0, 255).astype(np.uint8)

        combined = cv2.addWeighted(diff, 0.6, edge_diff, 0.4, 0)
        prob = combined.astype(np.float32) / 255.0
        return prob

    def auto_align_images(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        match_lighting: bool = True
    ) -> Tuple[Image.Image, Image.Image, np.ndarray]:
        """
        Co-register (align) bi-temporal satellite image pair.
        Preserves 1:1 geometry for pre-aligned pairs and uses bounded Rigid/Affine
        alignment for mismatched upload sizes, eliminating kaleidoscope distortion bugs.
        """
        from skimage.exposure import match_histograms

        np_a = np.array(img_a.convert('RGB'))
        np_b = np.array(img_b.convert('RGB'))

        h_a, w_a = np_a.shape[:2]
        h_b, w_b = np_b.shape[:2]

        # Case 1: Same dimensions -> Already georeferenced / orthorectified pair
        if (w_a, h_a) == (w_b, h_b):
            aligned_np_b = np_b
            valid_mask = np.ones((h_a, w_a), dtype=bool)
        else:
            # Case 2: Different dimensions -> Bounded Rigid Alignment (Scale + Rotation + Translation)
            aligned_np_b = None
            valid_mask = None

            try:
                gray_a = cv2.cvtColor(np_a, cv2.COLOR_RGB2GRAY)
                gray_b = cv2.cvtColor(np_b, cv2.COLOR_RGB2GRAY)

                detector = cv2.SIFT_create(nfeatures=2000)
                kp_a, des_a = detector.detectAndCompute(gray_a, None)
                kp_b, des_b = detector.detectAndCompute(gray_b, None)

                if des_a is not None and des_b is not None and len(kp_a) >= 8 and len(kp_b) >= 8:
                    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
                    matches = flann.knnMatch(des_b, des_a, k=2)
                    good = [m for m, n in matches if m.distance < 0.70 * n.distance]

                    if len(good) >= 8:
                        pts_b = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        pts_a = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                        # Estimate Rigid Euclidean / Affine transformation (NO projective kaleidoscope distortion)
                        M, inliers = cv2.estimateAffinePartial2D(pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if M is not None and inliers is not None and np.sum(inliers) >= 6:
                            scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                            # Strictly bound scale to plausible satellite zooms
                            if 0.6 <= scale <= 1.6:
                                warped_b = cv2.warpAffine(np_b, M, (w_a, h_a), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                ones_b = np.ones((h_b, w_b), dtype=np.uint8) * 255
                                v_mask = cv2.warpAffine(ones_b, M, (w_a, h_a), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                                valid_mask = (cv2.erode(v_mask, kernel) > 128)
                                aligned_np_b = warped_b
            except Exception as e:
                print(f"[InferenceEngine] Alignment notice: {e}")

            # Fallback if feature alignment was inconclusive
            if aligned_np_b is None:
                aligned_np_b = np.array(img_b.resize((w_a, h_a), Image.BILINEAR))
                valid_mask = np.ones((h_a, w_a), dtype=bool)

        # Radiometric Illumination Equalization
        if match_lighting and np.any(valid_mask):
            try:
                matched_b = match_histograms(aligned_np_b, np_a, channel_axis=-1)
                aligned_np_b = np.where(valid_mask[:, :, np.newaxis], matched_b, aligned_np_b).astype(np.uint8)
            except Exception as e:
                print(f"[InferenceEngine] Radiometric matching note: {e}")

        aligned_img_b = Image.fromarray(aligned_np_b)
        return img_a, aligned_img_b, valid_mask

    @torch.no_grad()
    def predict(
        self,
        img_a_input: Union[str, Image.Image, np.ndarray],
        img_b_input: Union[str, Image.Image, np.ndarray],
        threshold: Optional[float] = None,
        base_lat: float = 34.0837,
        base_lon: float = 74.7973,
        match_lighting: bool = True
    ) -> Dict[str, Any]:
        """
        Run change detection inference on bi-temporal satellite pair with automatic co-registration.
        """
        thresh = threshold if threshold is not None else self.threshold

        # Load images
        if isinstance(img_a_input, str):
            img_a = Image.open(img_a_input).convert('RGB')
        elif isinstance(img_a_input, np.ndarray):
            img_a = Image.fromarray(img_a_input).convert('RGB')
        else:
            img_a = img_a_input.convert('RGB')

        if isinstance(img_b_input, str):
            img_b = Image.open(img_b_input).convert('RGB')
        elif isinstance(img_b_input, np.ndarray):
            img_b = Image.fromarray(img_b_input).convert('RGB')
        else:
            img_b = img_b_input.convert('RGB')

        # Auto-align (co-register) image pair to eliminate false positives from size/crop shifts
        img_a, img_b, valid_mask = self.auto_align_images(img_a, img_b, match_lighting=match_lighting)
        orig_w, orig_h = img_a.size

        # PyTorch Model Forward Pass
        t1, t2, _ = self.preprocess(img_a, img_b, target_size=512)
        logits = self.model(t1, t2)
        probs_tensor = torch.sigmoid(logits).squeeze().cpu().numpy()

        # If trained weights are loaded, use neural predictions; otherwise blend with CV feature difference
        if self.has_trained_weights:
            prob_map = cv2.resize(probs_tensor, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            cv_prob = self._computer_vision_baseline(img_a, img_b)
            prob_map = cv2.resize(cv_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Zero out invalid boundary/non-overlapping margins
        if valid_mask.shape != prob_map.shape:
            valid_mask = cv2.resize(valid_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0

        prob_map[~valid_mask] = 0.0

        # Extract Satellite Engineered Features (VARI, EBI, Structural Gradients, Texture Entropy)
        img_a_np = np.array(img_a)
        img_b_np = np.array(img_b)
        engineered_features = self.feature_engineer.extract_bitemporal_features(img_a_np, img_b_np)

        # Feature-Assisted Decision Fusion (combines neural probabilities with physical built-up/gradient indices and vegetation suppression)
        delta_ebi_resized = cv2.resize(engineered_features['delta_ebi'], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        delta_grad_resized = cv2.resize(engineered_features['delta_grad'], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        veg_mask_resized = cv2.resize(engineered_features['veg_mask_b'].astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0

        prob_map = self.feature_engineer.fuse_predictions(
            prob_map,
            delta_ebi_resized,
            delta_grad_resized,
            veg_mask_resized
        )
        prob_map[~valid_mask] = 0.0

        # Binary thresholding
        raw_mask = (prob_map >= thresh).astype(np.uint8)

        # Extract grayscale representations for physical contrast verification
        np_a_rgb = np.array(img_a)
        np_b_rgb = np.array(img_b)
        gray_a = cv2.cvtColor(np_a_rgb, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(np_b_rgb, cv2.COLOR_RGB2GRAY)

        # Two-tier Hysteresis building extraction with solid roof filling
        high_t = max(0.25, thresh * 0.70)
        low_t = max(0.12, thresh * 0.35)
        clean_mask = self.postprocessor.clean_mask(
            raw_mask,
            prob_map=prob_map,
            img_a_gray=gray_a,
            img_b_gray=gray_b,
            high_thresh=high_t,
            low_thresh=low_t
        )
        clean_mask[~valid_mask] = 0

        # Cluster and Bounding Box Extraction
        clusters = self.postprocessor.extract_clusters(clean_mask, base_lat=base_lat, base_lon=base_lon)

        # Threat Assessment
        total_pixels = int(np.sum(valid_mask)) if np.any(valid_mask) else (orig_w * orig_h)
        change_pixels = int(np.sum(clean_mask > 0))
        threat_info = self.postprocessor.assess_threat_level(total_pixels, change_pixels, clusters)

        # Generate Visual Artifacts
        tactical_overlay = self.postprocessor.generate_tactical_overlay(img_b_np, clean_mask, clusters)
        heatmap_rgb = self.postprocessor.generate_heatmap(prob_map, target_shape=(orig_h, orig_w))

        return {
            'prob_map': prob_map,
            'raw_mask': raw_mask,
            'clean_mask': clean_mask,
            'tactical_overlay': tactical_overlay,
            'heatmap_rgb': heatmap_rgb,
            'clusters': clusters,
            'threat_info': threat_info,
            'engineered_features': engineered_features,
            'has_trained_weights': self.has_trained_weights
        }
