# ResGatedUNet-Model

## Boundary-Aware Lunar Crater Segmentation

This repository provides the implementation and supporting resources for the research work:

**"ResGatedUNet: A Boundary-Aware Lunar Crater Segmentation Network from DEM Data"**

The repository is intended to support **transparency, reproducibility, and evaluation** of the proposed lunar crater segmentation and detection framework.

---

## 📌 Contents

The repository provides:

- ResGatedUNet model implementation
- Preprocessing scripts
- Geographical train/validation/test split definition
- Trained model weights
- Multi-scale circular template matching (MCTM) code
- Duplicate suppression
- Pixel-, boundary-, and crater-level evaluation code

---

## 📊 Dataset

The experiments use lunar DEM data covering approximately **60°S–60°N**.

The dataset contains:

- **30,000** training patches
- **3,000** validation patches
- **3,000** test patches
- Patch size: **256 × 256 pixels**
- Single-channel DEM input

The geographical split is defined by longitude:

| Split | Longitude |
|---|---|
| Training | −180° to −60° |
| Validation | −60° to +60° |
| Test | +60° to +180° |

The original DEM and crater catalogue are not included in this repository.

---

## 🧩 Preprocessing

The patch-generation procedure is derived from the **DeepMoon** framework and adapted to the DEM and crater catalogue used in this study.

The preprocessing pipeline includes:

- DEM patch generation
- Crater mask generation
- DEM normalization
- Boundary/edge-map generation

DeepMoon repository:

https://github.com/silburt/DeepMoon

---

## 🧠 Model

The proposed **ResGatedUNet** uses a ResNet-50-based encoder, gated spatial connections, and an auxiliary edge-learning branch.

The model produces two output maps:

1. Crater segmentation map
2. Edge map

The segmentation map is used for crater detection using multi-scale circular template matching.

---

## 🎯 Crater Detection

Crater candidates are obtained using **Multi-Scale Circular Template Matching (MCTM)** followed by radius-aware duplicate suppression.

The main parameters used in the experiments are:

- NCC threshold: **0.50**
- Radius range: **3–100 pixels**
- Radius step: **2 pixels**
- Ring thickness: **2 pixels**
- Local maximum distance: **5 pixels**

---

## 📈 Evaluation

The repository provides evaluation at three levels:

### Pixel level
- Precision
- Recall
- F1-score
- IoU
- Dice

### Boundary level
- Boundary Precision
- Boundary Recall
- Boundary F1-score
- Average Contour Distance (ACD)
- Hausdorff Distance (HD)

### Crater level
- Crater Precision
- Crater Recall
- Crater F1-score
- Center error
- Radius error
- Diameter error

---

## ⚖️ Trained Weights

The trained ResGatedUNet weights are provided in:

```text
weights/ResGatedUNet_weights.pth
---

## 📬 Contact

For any queries, please contact:

**Deeksha Sahu**  
Ph.D. Research Scholar  
National Institute of Technology Raipur, India  

Email: *deeksha.phd2022.etc@nitrr.ac.in*

---

## 📖 Citation

If you use this dataset, please cite:
