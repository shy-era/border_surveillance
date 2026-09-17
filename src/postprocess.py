import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class BorderSurveillancePostProcessor:
    """
    Post-processing, Noise Reduction, Cluster Detection, and Threat Alert Logic
    for Satellite Border Change Detection.
    """
    def __init__(
        self,
        min_cluster_area: int = 15,
        morph_kernel_size: int = 3,
        gsd_meters_per_pixel: float = 0.5
    ):
        self.min_cluster_area = min_cluster_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        self.gsd = gsd_meters_per_pixel

    def clean_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening & closing to eliminate speckle noise and fill structural holes.
        """
        mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255

        # Morphological Opening (remove isolated pixels)
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, self.kernel, iterations=1)
        # Morphological Closing (bridge structural gaps)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        # Filter out clusters below minimum area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        filtered = np.zeros_like(cleaned)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_cluster_area:
                filtered[labels == i] = 255

        return (filtered > 0).astype(np.uint8)

    def extract_clusters(self, clean_mask: np.ndarray, base_lat: float = 34.0837, base_lon: float = 74.7973) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes, centroids, area metrics, and simulated geocoordinates.
        """
        contours, _ = cv2.findContours(clean_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clusters = []

        h, w = clean_mask.shape

        for idx, cnt in enumerate(contours):
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_cluster_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + bw // 2, y + bh // 2

            # Ground area in square meters
            area_sqm = area_px * (self.gsd ** 2)

            # Simulated GPS coordinates (approximate 0.00001 deg per meter)
            meter_offset_x = (cx - w / 2) * self.gsd
            meter_offset_y = (cy - h / 2) * self.gsd
            cluster_lat = base_lat + (meter_offset_y / 111139.0)
            cluster_lon = base_lon + (meter_offset_x / (111139.0 * np.cos(np.radians(base_lat))))

            clusters.append({
                'id': f"TGT-{idx+1:03d}",
                'bbox': (int(x), int(y), int(bw), int(bh)),
                'centroid': (int(cx), int(cy)),
                'area_px': int(area_px),
                'area_sqm': round(float(area_sqm), 1),
                'lat': round(float(cluster_lat), 6),
                'lon': round(float(cluster_lon), 6),
                'perimeter': round(float(cv2.arcLength(cnt, True)), 1)
            })

        # Sort clusters by area descending
        clusters.sort(key=lambda c: c['area_px'], reverse=True)
        return clusters

    def assess_threat_level(self, total_pixels: int, change_pixels: int, clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate operational threat level for defense/surveillance.
        """
        pct_change = (change_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        num_clusters = len(clusters)
        max_cluster_area = clusters[0]['area_sqm'] if clusters else 0.0
        total_sqm_changed = (change_pixels * (self.gsd ** 2))

        if pct_change < 0.15 and num_clusters == 0:
            level = "SECURE / NORMAL"
            color = "#00E676"  # Bright Green
            defcon = "DEFCON 5 - All Clear"
            action = "Routine automated surveillance. No anomalous structural activity detected."
            threat_score = 10
        elif pct_change < 1.0 and max_cluster_area < 80:
            level = "ELEVATED (LOW)"
            color = "#FFD600"  # Amber
            defcon = "DEFCON 4 - Low Advisory"
            action = "Minor anomaly detected (possible vehicle track, tent, or small earthwork). Log for daily review."
            threat_score = 35
        elif pct_change < 4.0 or (num_clusters >= 3 and max_cluster_area >= 80):
            level = "HIGH ALERT (MEDIUM)"
            color = "#FF6D00"  # Orange
            defcon = "DEFCON 3 - Tactical Caution"
            action = "Significant structural expansion or multiple outposts detected. Dispatch aerial drone / patrol verification."
            threat_score = 70
        else:
            level = "CRITICAL THREAT (HIGH)"
            color = "#FF1744"  # Neon Red
            defcon = "DEFCON 2 - Hostile Activity / Fortification"
            action = "Major infrastructure or military outpost detected in border buffer zone. Immediate tactical escalation required!"
            threat_score = 95

        return {
            'threat_level': level,
            'color': color,
            'defcon': defcon,
            'action_recommendation': action,
            'threat_score': threat_score,
            'pct_change': round(pct_change, 3),
            'total_sqm_changed': round(total_sqm_changed, 1),
            'num_clusters': num_clusters,
            'max_cluster_area_sqm': max_cluster_area
        }

    def generate_tactical_overlay(
        self,
        image: np.ndarray,
        clean_mask: np.ndarray,
        clusters: List[Dict[str, Any]],
        draw_boxes: bool = True
    ) -> np.ndarray:
        """
        Draw translucent change highlight, bounding boxes, target IDs, and crosshairs.
        """
        h, w = image.shape[:2]
        overlay = image.copy()

        # Ensure mask spatial dimensions match image exactly
        if clean_mask.shape[:2] != (h, w):
            clean_mask = cv2.resize(clean_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # Robust red highlight on changed regions
        mask_binary = (clean_mask > 0).astype(np.uint8)
        if np.any(mask_binary):
            # Create red highlight overlay in RGB/BGR
            red_tint = np.zeros_like(image)
            red_tint[:, :, 0] = 255  # Red
            red_tint[:, :, 1] = 50   # Orange-red
            red_tint[:, :, 2] = 0

            # Alpha blend where mask is positive
            blended = cv2.addWeighted(overlay, 0.65, red_tint, 0.35, 0)
            mask_3d = mask_binary[:, :, np.newaxis] == 1
            overlay = np.where(mask_3d, blended, overlay)

        if draw_boxes:
            for c in clusters:
                x, y, bw, bh = c['bbox']
                cx, cy = c['centroid']
                tid = c['id']

                # Clamp bounding box inside image boundaries
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                # Bounding box
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 255), 2)  # Cyan/Yellow

                # Target ID Label badge
                label = f"{tid}: {c['area_sqm']}m²"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(overlay, (x, max(0, y - 18)), (x + tw + 6, max(0, y)), (0, 0, 0), -1)
                cv2.putText(overlay, label, (x + 3, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

                # Centroid Crosshair
                cx = max(0, min(cx, w - 1))
                cy = max(0, min(cy, h - 1))
                cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

        return overlay

    def generate_heatmap(self, prob_map: np.ndarray, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert probability map (0.0 to 1.0) into RGB Jet Heatmap.
        """
        if target_shape and prob_map.shape[:2] != target_shape:
            prob_map = cv2.resize(prob_map, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

        norm_map = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
