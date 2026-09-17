# 🛰️ AEGIS-SAT: Satellite-Based Border Surveillance & Structural Change Detection

An end-to-end Deep Learning and Computer Vision system designed for **bi-temporal satellite border surveillance**. It identifies unauthorized structural encroachment, outpost construction, road clearing, and fortified installations across border buffer zones.

---

## 🏗️ System Architecture

```
Bi-temporal Satellite Passes
  ┌──────────────┐
  │ Image T1     │ ──► [ ResNet-34 Encoder (Shared) ] ──┐
  │ (Baseline)   │                                      ├──► [ Multi-Scale Feature Fusion ] ──► [ U-Net Decoder ] ──► Probability Map
  │ Image T2     │ ──► [ ResNet-34 Encoder (Shared) ] ──┘          |F1 - F2| + Concat
  │ (Surveillance│
  └──────────────┘
                                                                                                               │
                                                                                                               ▼
                                                                                                  [ Morphological Filtering ]
                                                                                                               │
                                                                                                               ▼
                                                                                                  [ Connected Component Clustering ]
                                                                                                               │
                                                                                                               ▼
                                                                                                  [ Tactical Threat Assessment & HUD ]
```

---

## ⚡ Key Highlights

- **Siamese ResNet Backbone (`resnet18` / `resnet34`)**: Weight-shared dual encoders that extract multi-scale semantic feature pyramids ($C_0, \dots, C_4$).
- **Multi-Scale Feature Difference Fusion**: Combines absolute difference $|F_1 - F_2|$ and concatenation $[F_1, F_2, |F_1 - F_2|]$ to capture both subtle textural changes and large structural formations.
- **Combined BCE + Dice Loss**: Mitigates extreme class imbalance where structural changes occupy $< 5\%$ of satellite pixels.
- **Surveillance Threat Engine**:
  - **Morphological noise suppression** to eliminate sensor artifacts and seasonal foliage variations.
  - **Automated target extraction**: Bounding boxes, centroid crosshairs, area ($m^2$), and simulated GPS coordinates.
  - **DEFCON Threat Leveling**: Automatically assigns threat tiers (`DEFCON 5 Normal` to `DEFCON 2 Critical Threat`).
- **Tactical Streamlit Surveillance Dashboard**:
  - Preloaded border simulation sectors (Outpost, Watchtower, Road Intrusion, River Checkpoint, Control Sector).
  - Manual upload for custom satellite pairs.
  - Live ESRI Satellite Map integration via Folium.
  - 1-click CSV target manifest and incident report export.

---

## 📂 Project Structure

```
border_2/
├── data/
│   ├── samples/                # Preloaded border sector pairs & masks
│   │   ├── A/                  # Time T1 images (Before)
│   │   ├── B/                  # Time T2 images (After)
│   │   └── label/              # Binary change masks
│   └── LEVIR-CD/               # Full benchmark dataset (optional)
├── src/
│   ├── __init__.py
│   ├── model.py                # Siamese ResNet U-Net architecture
│   ├── dataset.py              # Bi-temporal dataset loader & augmentations
│   ├── losses.py               # Combined BCE + Dice & Focal Loss
│   ├── metrics.py              # IoU, F1, Precision, Recall, OA
│   ├── postprocess.py          # Morphological filter, clustering, DEFCON threat engine
│   ├── infer.py                # Inference pipeline
│   ├── train.py                # Training loop with validation & checkpointing
│   └── generate_samples.py     # Synthetic border scenario generator
├── gui/
│   └── app.py                  # Tactical Streamlit Defense UI
├── checkpoints/                # Saved model weights
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Demo Border Scenarios
```bash
python -m src.generate_samples
```

### 3. Launch the Tactical Surveillance Dashboard
```bash
streamlit run gui/app.py
```

### 4. Train on LEVIR-CD or Custom Dataset
```bash
# Train with ResNet-18 or ResNet-34 encoder
python -m src.train --data_dir data/samples --epochs 15 --batch_size 4 --backbone resnet18
```

---

## 🎓 Viva & Presentation Talking Points

1. **Why Siamese Network?**
   - Direct image subtraction $(I_1 - I_2)$ fails due to differences in sunlight angles, seasonal vegetation, and sensor noise.
   - Siamese feature extractors map images into deep semantic representations where structural changes are isolated from atmospheric noise.
2. **Why BCE + Dice Loss?**
   - Binary Cross Entropy (BCE) treats every pixel equally, which biases the model towards predicting all-zeros when changed pixels are $< 5\%$.
   - Dice Loss directly optimizes the overlap (F1-score), forcing the network to delineate small structure boundaries accurately.
3. **What is GSD (Ground Sample Distance)?**
   - Ground resolution in meters per pixel (e.g., $0.5\text{m/px}$). It allows accurate real-world area calculation $(\text{Area}_{m^2} = \text{pixels} \times \text{GSD}^2)$.
