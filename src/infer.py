import os
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union, Tuple
import cv2

from src.model import SiameseResNetUNet, create_model
from src.postprocess import BorderSurveillancePostProcessor


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

    def preprocess(self, img_a: Image.Image, img_b: Image.Image, target_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
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

    @torch.no_grad()
    def predict(
        self,
        img_a_input: Union[str, Image.Image, np.ndarray],
        img_b_input: Union[str, Image.Image, np.ndarray],
        threshold: Optional[float] = None,
        base_lat: float = 34.0837,
        base_lon: float = 74.7973
    ) -> Dict[str, Any]:
        """
        Run change detection inference on bi-temporal satellite pair.
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

        # Auto-align bi-temporal pair dimensions if different
        if img_b.size != img_a.size:
            img_b = img_b.resize(img_a.size, Image.BILINEAR)

        orig_w, orig_h = img_a.size

        # PyTorch Model Forward Pass
        t1, t2, _ = self.preprocess(img_a, img_b, target_size=256)
        logits = self.model(t1, t2)
        probs_tensor = torch.sigmoid(logits).squeeze().cpu().numpy()

        # If trained weights are loaded, use neural predictions; otherwise blend with CV feature difference for demo stability
        if self.has_trained_weights:
            prob_map = cv2.resize(probs_tensor, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            cv_prob = self._computer_vision_baseline(img_a, img_b)
            prob_map = cv2.resize(cv_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Binary thresholding
        raw_mask = (prob_map >= thresh).astype(np.uint8)

        # Morphological postprocessing
        clean_mask = self.postprocessor.clean_mask(raw_mask)

        # Cluster and Bounding Box Extraction
        clusters = self.postprocessor.extract_clusters(clean_mask, base_lat=base_lat, base_lon=base_lon)

        # Threat Assessment
        total_pixels = orig_w * orig_h
        change_pixels = int(np.sum(clean_mask > 0))
        threat_info = self.postprocessor.assess_threat_level(total_pixels, change_pixels, clusters)

        # Generate Visual Artifacts
        img_b_np = np.array(img_b)
        tactical_overlay = self.postprocessor.generate_tactical_overlay(img_b_np, clean_mask, clusters)
        heatmap_rgb = self.postprocessor.generate_heatmap(prob_map)

        return {
            'prob_map': prob_map,
            'raw_mask': raw_mask,
            'clean_mask': clean_mask,
            'tactical_overlay': tactical_overlay,
            'heatmap_rgb': heatmap_rgb,
            'clusters': clusters,
            'threat_info': threat_info,
            'has_trained_weights': self.has_trained_weights
        }
