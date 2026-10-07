import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple


class SatelliteFeatureEngineer:
    """
    Feature Engineering Module for Bi-Temporal Satellite Surveillance.
    Extracts classical remote sensing, spectral, structural gradient, and texture features
    to complement deep learning change detection.
    """
    def __init__(self):
        pass

    def compute_vari(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Visible Atmospherically Resistant Index (VARI) - proxy for NDVI using RGB channels.
        Formula: (Green - Red) / (Green + Red - Blue + eps)
        Range: -1.0 to 1.0 (Higher values = dense vegetation; Lower = concrete/built-up structures).
        """
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)

        denom = g + r - b
        denom[denom == 0] = 1e-5

        vari = (g - r) / denom
        return np.clip(vari, -1.0, 1.0)

    def compute_structural_gradient(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Computes Sobel edge gradient magnitude to identify structural boundaries,
        road alignments, and building perimeters.
        """
        sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return grad_norm

    def compute_texture_entropy(self, img_gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
        """
        Local standard deviation / texture entropy filter.
        Differentiates between smooth natural terrain (grass, soil, water)
        and high-frequency man-made structures (concrete roofs, gravel outposts, fences).
        """
        img_f = img_gray.astype(np.float32)
        mean = cv2.blur(img_f, (kernel_size, kernel_size))
        mean_sq = cv2.blur(img_f**2, (kernel_size, kernel_size))
        variance = np.maximum(0, mean_sq - mean**2)
        std_dev = np.sqrt(variance)
        return cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def compute_builtup_index_proxy(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Engineered Built-up Index (EBI) proxy for RGB satellite imagery.
        Formula: (Red + Blue - 2*Green) / (Red + Blue + 2*Green + eps)
        High values indicate artificial surfaces (roofs, concrete, dry earthworks).
        """
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)

        numerator = (r + b) - (2.0 * g)
        denominator = (r + b) + (2.0 * g) + 1e-5
        ebi = numerator / denominator
        return np.clip(ebi, -1.0, 1.0)

    def calibrate_probabilities(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Dynamically calibrate neural probability by removing background resting baseline shift.
        Scales the dynamic contrast range from 0.0 (background) to 1.0 (true structural change).
        """
        if prob_map.size == 0:
            return prob_map
        # Estimate background baseline from the lower 20th percentile
        baseline = float(np.percentile(prob_map, 20))
        calibrated = np.clip((prob_map - baseline) / (1.0 - baseline + 1e-5), 0.0, 1.0)
        return calibrated

    def fuse_predictions(
        self,
        neural_prob: np.ndarray,
        builtup_score: np.ndarray,
        delta_grad: np.ndarray,
        veg_mask: np.ndarray
    ) -> np.ndarray:
        """
        Feature-Assisted Decision Fusion:
        1. Calibrates raw neural probabilities to remove background baseline shift.
        2. Blends with physical Built-Up Concrete Index (ΔEBI) and Sobel Structural Gradients (ΔSobel).
        3. Strictly suppresses all tree canopies, grass, and foliage using Vegetation Invariant Mask.
        """
        calib_prob = self.calibrate_probabilities(neural_prob)
        norm_grad = delta_grad.astype(np.float32) / 255.0

        # Multi-modal fusion
        fused = 0.50 * calib_prob + 0.35 * builtup_score + 0.15 * norm_grad

        # STRICT VEGETATION SUPPRESSION: Zero out all tree canopies, leaves, and green foliage
        fused[veg_mask] = 0.0
        return np.clip(fused, 0.0, 1.0)

    def extract_bitemporal_features(self, img_a_np: np.ndarray, img_b_np: np.ndarray) -> Dict[str, Any]:
        """
        Extract complete engineered feature suite between pre-change (T1) and post-change (T2).
        Strictly isolates artificial building construction from natural vegetation shifts.
        """
        gray_a = cv2.cvtColor(img_a_np, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b_np, cv2.COLOR_RGB2GRAY)

        # 1. Vegetation Indices (VARI)
        vari_a = self.compute_vari(img_a_np)
        vari_b = self.compute_vari(img_b_np)
        delta_vari = np.clip(vari_a - vari_b, 0.0, 1.0)

        # Vegetation Mask: Pixels in Time-2 that are active green tree foliage / grass
        veg_mask_b = (vari_b > 0.10)

        # 2. Engineered Built-Up Concrete Index (EBI)
        ebi_a = self.compute_builtup_index_proxy(img_a_np)
        ebi_b = self.compute_builtup_index_proxy(img_b_np)
        delta_ebi = np.clip(ebi_b - ebi_a, 0.0, 1.0)
        # Suppress built-up index on trees
        delta_ebi[veg_mask_b] = 0.0

        # 3. Structural Gradient & Edge Emergence (Sobel)
        grad_a = self.compute_structural_gradient(gray_a)
        grad_b = self.compute_structural_gradient(gray_b)
        delta_grad = cv2.absdiff(grad_b, grad_a)
        delta_grad[veg_mask_b] = 0

        # 4. Local Texture Entropy Shift
        tex_a = self.compute_texture_entropy(gray_a)
        tex_b = self.compute_texture_entropy(gray_b)
        delta_tex = cv2.absdiff(tex_b, tex_a)
        delta_tex[veg_mask_b] = 0

        # 5. Composite Built-Up Construction Score (0.0 to 1.0)
        norm_grad = delta_grad.astype(np.float32) / 255.0
        norm_tex = delta_tex.astype(np.float32) / 255.0
        composite_score = (
            0.55 * delta_ebi +
            0.30 * norm_grad +
            0.15 * norm_tex
        )
        composite_score[veg_mask_b] = 0.0
        composite_score = np.clip(composite_score, 0.0, 1.0)

        # Generate RGB Colormaps for UI Visualization
        heatmap_vari = cv2.applyColorMap((delta_vari * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        heatmap_grad = cv2.applyColorMap(delta_grad, cv2.COLORMAP_MAGMA)
        heatmap_ebi = cv2.applyColorMap((delta_ebi * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        heatmap_comp = cv2.applyColorMap((composite_score * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Summary Statistical Metrics
        metrics = {
            "mean_veg_loss_index": round(float(np.mean(delta_vari)), 4),
            "mean_builtup_emergence_index": round(float(np.mean(delta_ebi)), 4),
            "mean_structural_gradient": round(float(np.mean(delta_grad)), 2),
            "mean_texture_entropy_shift": round(float(np.mean(delta_tex)), 2),
            "structural_anomaly_ratio": round(float(np.sum(composite_score > 0.35) / max(1, composite_score.size) * 100), 2)
        }

        return {
            "delta_vari": delta_vari,
            "delta_ebi": delta_ebi,
            "delta_grad": delta_grad,
            "delta_tex": delta_tex,
            "veg_mask_b": veg_mask_b,
            "composite_score": composite_score,
            "vis_vari_rgb": cv2.cvtColor(heatmap_vari, cv2.COLOR_BGR2RGB),
            "vis_grad_rgb": cv2.cvtColor(heatmap_grad, cv2.COLOR_BGR2RGB),
            "vis_ebi_rgb": cv2.cvtColor(heatmap_ebi, cv2.COLOR_BGR2RGB),
            "vis_comp_rgb": cv2.cvtColor(heatmap_comp, cv2.COLOR_BGR2RGB),
            "metrics": metrics
        }
