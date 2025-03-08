# KARMA: Efficient Structural Defect Segmentation via Kolmogorov-Arnold Representation Learning

KARMA (Kolmogorov-Arnold Representation Mapping Architecture) is a highly efficient semantic segmentation framework designed specifically for structural defect detection in civil infrastructure. It models complex defect patterns through compositions of one-dimensional functions rather than conventional convolutions, resulting in significantly reduced parameters while maintaining or improving accuracy.

Official implementation of the paper "KARMA: Efficient Structural Defect Segmentation via Kolmogorov-Arnold Representation Learning" submitted to ICCV 2025.

## Key Innovations

- **Parameter-efficient TiKAN module**: Leverages low-rank factorization for KAN-based feature transformation, dramatically reducing parameter count
- **Optimized feature pyramid structure**: Uses separable convolutions for multi-scale defect analysis with minimal computational overhead
- **Static-dynamic prototype mechanism**: Enhances feature representation for imbalanced defect classes, improving rare defect detection

## Performance Highlights

- **High Accuracy**: Outperforms state-of-the-art approaches by 5-10% in mean IoU on structural defect datasets
- **Parameter Efficiency**: Uses only 0.959M parameters (97% fewer than comparable models)
- **Computational Efficiency**: Operates at 0.264 GFLOPS, enabling real-time deployment in automated inspection systems

## Model Architecture

KARMA integrates Kolmogorov-Arnold modules into an adaptive feature pyramid network:
- Bottom-up pathway with InceptionSepConv blocks for efficient multi-scale feature extraction
- TiKAN enhancement at the deepest feature level
- Top-down pathway with feature fusion
- Multi-scale prediction heads for comprehensive defect analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/faeyelab/karma.git
cd karma

# Create conda environment (optional)
conda create -n karma python=3.8
conda activate karma

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config config/default.yaml
```

### Inference

```python
import torch
from models.karma import Karma

# Load model
model = Karma(num_classes=7)
model.load_state_dict(torch.load('path/to/weights.pth'))
model.eval()

# Inference
with torch.no_grad():
    input_tensor = torch.randn(1, 3, 256, 256)  # Example input
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
```

## Datasets

KARMA is evaluated on two challenging structural defect datasets:

### Structural Defects Dataset (S2DS)
Contains 743 high-resolution images of concrete surfaces with pixel-wise annotations across seven distinct classes:
- Class 0: Background
- Class 1: Crack (linear fractures)
- Class 2: Spalling (surface detachment)
- Class 3: Corrosion (rust)
- Class 4: Efflorescence (chemical deposits)
- Class 5: Vegetation (plant growth)
- Class 6: Control Point (fiducial markers)

#### S2DS Download
The S2DS dataset can be downloaded from:
- [Google Drive Link](https://drive.google.com/file/d/1PQ50QKfy2vnDOHSmw5bpBFi33hZsSXuM/view?usp=sharing)

After downloading, extract the dataset and organize it with the following structure:
```
data/
├── s2ds/
    ├── train/
    ├── val/
    ├── test/
```

### Culvert-Sewer Defects Dataset (CSDD)
Comprises approximately 6,300 frames from 580 annotated underground inspection videos covering eight defect classes:
- Cracks
- Roots
- Holes
- Joint issues
- Deformation
- Fracture
- Encrustation/deposits
- Loose gasket

## Project Structure

```
karma/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── config/
│   └── default.yaml
├── data/
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── karma.py
│   ├── layers.py
│   ├── blocks.py
│   └── tikan.py
├── losses/
│   ├── __init__.py
│   └── losses.py
├── metrics/
│   ├── __init__.py
│   └── metrics.py
├── utils/
│   ├── __init__.py
│   └── utils.py
└── train.py
```

## Citation

If you use this code in your research, please cite our paper:

```
@article{karma2025,
  title={KARMA: Efficient Structural Defect Segmentation via Kolmogorov-Arnold Representation Learning},
  author={[Anonymous]},
  journal={Submitted to ICCV},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
