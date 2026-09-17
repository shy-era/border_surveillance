import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_realistic_terrain(width: int, height: int, terrain_type: str = 'desert') -> np.ndarray:
    """Generate textured synthetic satellite background terrain."""
    base = np.zeros((height, width, 3), dtype=np.uint8)

    # Perlin-like noise using multi-scale blur
    noise1 = np.random.randint(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
    noise1 = cv2.resize(noise1, (width, height), interpolation=cv2.INTER_CUBIC)

    noise2 = np.random.randint(0, 255, (height // 2, width // 2, 3), dtype=np.uint8)
    noise2 = cv2.resize(noise2, (width, height), interpolation=cv2.INTER_LINEAR)

    blend = cv2.addWeighted(noise1, 0.7, noise2, 0.3, 0)

    if terrain_type == 'desert':
        # Sandy ochre / tan shades
        color_base = np.array([195, 175, 140], dtype=np.float32)
        variance = (blend.astype(np.float32) - 128) * 0.25
        res = np.clip(color_base + variance, 0, 255).astype(np.uint8)
    elif terrain_type == 'mountain':
        # Rocky slate / grey-brown shades
        color_base = np.array([120, 115, 110], dtype=np.float32)
        variance = (blend.astype(np.float32) - 128) * 0.4
        res = np.clip(color_base + variance, 0, 255).astype(np.uint8)
    elif terrain_type == 'forest':
        # Olive green / dark forest canopy
        color_base = np.array([65, 95, 50], dtype=np.float32)
        variance = (blend.astype(np.float32) - 128) * 0.3
        res = np.clip(color_base + variance, 0, 255).astype(np.uint8)
    elif terrain_type == 'river':
        # Desert with river channel
        color_base = np.array([180, 160, 130], dtype=np.float32)
        variance = (blend.astype(np.float32) - 128) * 0.2
        res = np.clip(color_base + variance, 0, 255).astype(np.uint8)
        # Draw river
        pts = np.array([[int(width * 0.3), 0], [int(width * 0.45), height // 2], [int(width * 0.4), height]], np.int32)
        cv2.polylines(res, [pts], False, (140, 100, 60), 36)  # BGR water/dark blue
    else:
        res = blend

    return res


def add_satellite_artifacts(img: np.ndarray, light_shift: int = 0) -> np.ndarray:
    """Add subtle atmospheric haze and sensor noise."""
    res = img.astype(np.float32) + light_shift
    sensor_noise = np.random.normal(0, 3, img.shape)
    res = np.clip(res + sensor_noise, 0, 255).astype(np.uint8)
    return res


def generate_all_samples(output_dir: str = 'data/samples'):
    """Generate preset scenario pairs and masks."""
    os.makedirs(os.path.join(output_dir, 'A'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'B'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'label'), exist_ok=True)

    size = 256

    scenarios = [
        {
            'name': 'sector_alpha_outpost',
            'title': 'Sector Alpha - New Fortified Border Outpost',
            'terrain': 'desert',
            'lat': 34.1205,
            'lon': 74.8320,
            'description': 'Detection of two concrete bunker structures and perimeter trench.',
            'add_structures': [
                {'type': 'rect', 'box': (60, 70, 50, 40), 'color': (220, 220, 225)},
                {'type': 'rect', 'box': (140, 80, 60, 55), 'color': (210, 210, 215)},
                {'type': 'rect', 'box': (100, 160, 45, 35), 'color': (190, 190, 195)},
            ]
        },
        {
            'name': 'sector_bravo_watchtower_helipad',
            'title': 'Sector Bravo - Watchtower & Staging Pad',
            'terrain': 'mountain',
            'lat': 34.2541,
            'lon': 74.9182,
            'description': 'Detection of an elevated observation post and circular landing strip.',
            'add_structures': [
                {'type': 'rect', 'box': (80, 100, 40, 40), 'color': (230, 230, 230)},
                {'type': 'circle', 'center': (180, 150), 'radius': 28, 'color': (180, 185, 190)},
            ]
        },
        {
            'name': 'sector_charlie_road_depot',
            'title': 'Sector Charlie - Illegal Access Road & Depot',
            'terrain': 'forest',
            'lat': 33.9850,
            'lon': 74.6521,
            'description': 'Forest clearing with newly laid vehicle track and logistics barracks.',
            'add_structures': [
                {'type': 'line', 'pt1': (20, 40), 'pt2': (230, 210), 'thickness': 12, 'color': (160, 150, 130)},
                {'type': 'rect', 'box': (150, 130, 55, 45), 'color': (225, 225, 230)},
            ]
        },
        {
            'name': 'sector_delta_river_checkpoint',
            'title': 'Sector Delta - Riverbank Pier & Barracks',
            'terrain': 'river',
            'lat': 34.0512,
            'lon': 74.7745,
            'description': 'Riverine checkpoint construction with dock extension.',
            'add_structures': [
                {'type': 'rect', 'box': (120, 90, 40, 60), 'color': (220, 215, 210)},
                {'type': 'rect', 'box': (170, 110, 30, 25), 'color': (200, 200, 205)},
            ]
        },
        {
            'name': 'sector_echo_secure_control',
            'title': 'Sector Echo - Secure Ridge (Zero Encroachment Control)',
            'terrain': 'mountain',
            'lat': 34.3100,
            'lon': 75.0120,
            'description': 'Baseline border sector showing natural seasonal lighting without structural intrusion.',
            'add_structures': []
        }
    ]

    for sc in scenarios:
        name = sc['name']
        t1_base = create_realistic_terrain(size, size, sc['terrain'])
        t1 = add_satellite_artifacts(t1_base, light_shift=0)

        t2_base = t1_base.copy()
        mask = np.zeros((size, size), dtype=np.uint8)

        # Draw structural changes
        for struct in sc['add_structures']:
            stype = struct['type']
            col = struct['color']
            if stype == 'rect':
                x, y, w, h = struct['box']
                cv2.rectangle(t2_base, (x, y), (x + w, y + h), col, -1)
                # Shadow
                cv2.rectangle(t2_base, (x + w, y + 5), (x + w + 8, y + h + 8), (40, 40, 40), -1)
                # Mark ground truth mask
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            elif stype == 'circle':
                cx, cy = struct['center']
                r = struct['radius']
                cv2.circle(t2_base, (cx, cy), r, col, -1)
                cv2.circle(mask, (cx, cy), r, 255, -1)
            elif stype == 'line':
                p1 = struct['pt1']
                p2 = struct['pt2']
                th = struct['thickness']
                cv2.line(t2_base, p1, p2, col, th)
                cv2.line(mask, p1, p2, 255, th)

        t2 = add_satellite_artifacts(t2_base, light_shift=8)  # slight sun angle difference

        # Save images
        cv2.imwrite(os.path.join(output_dir, 'A', f"{name}.png"), cv2.cvtColor(t1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, 'B', f"{name}.png"), cv2.cvtColor(t2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, 'label', f"{name}.png"), mask)

    print(f"Successfully generated {len(scenarios)} realistic demo scenarios in {output_dir}")


if __name__ == '__main__':
    generate_all_samples()
