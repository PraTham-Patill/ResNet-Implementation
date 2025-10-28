# ResNet-56 Implementation on CIFAR-10

A PyTorch implementation of ResNet-56 for CIFAR-10 classification, following the original paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) by Kaiming He et al.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training Details](#training-details)
- [References](#references)

## 🔍 Overview

This project implements the ResNet-56 architecture from scratch using PyTorch, trained on the CIFAR-10 dataset. The implementation closely follows the specifications from the original paper, including:

- Residual blocks with skip connections
- Batch normalization
- Weight initialization (Kaiming initialization)
- Data augmentation (random cropping, horizontal flipping)
- Learning rate scheduling with warm restarts
- Comprehensive training and validation pipeline

## 🏗️ Architecture

ResNet-56 consists of:
- **Input**: 32×32×3 CIFAR-10 images
- **Initial Convolution**: 3×3 conv, 16 filters
- **Residual Blocks**: 
  - Stage 1: 9 blocks, 16 filters, 32×32 feature maps
  - Stage 2: 9 blocks, 32 filters, 16×16 feature maps (stride 2)
  - Stage 3: 9 blocks, 64 filters, 8×8 feature maps (stride 2)
- **Global Average Pooling**: 8×8 → 1×1
- **Fully Connected**: 64 → 10 classes

**Total Parameters**: ~855K parameters

### Residual Block Structure
```
    Input
      |
      ├─────────────────┐
      |                 |
   [3×3 conv]          |
      |                 |
 [Batch Norm]          |
      |                 |
    [ReLU]             |
      |                 |
   [3×3 conv]          |
      |                 |
 [Batch Norm]          |
      |                 |
      └────[Add]────────┘
             |
          [ReLU]
             |
          Output
```

## 📊 Results

- **Test Accuracy**: ~93.5% (target as per paper)
- **Training Time**: ~4-6 hours on GPU
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate Schedule**: Step decay at 32k and 48k iterations

### Training Curves
Training and validation metrics are logged during training. See `analysis/` folder for visualization outputs.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/PraTham-Patill/ResNet-Implementation.git
cd ResNet-Implementation

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training from Scratch
```bash
python src/train_resnet56_cifar10.py
```

The script will:
1. Download CIFAR-10 dataset automatically (if not present)
2. Train ResNet-56 for 205 epochs
3. Save checkpoints and best model
4. Generate training plots and analysis

### Configuration
Key hyperparameters in `src/train_resnet56_cifar10.py`:
```python
BATCH_SIZE = 128
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TOTAL_EPOCHS = 205
```

## 📁 Project Structure

```
ResNet-Implementation/
├── src/
│   └── train_resnet56_cifar10.py  # Main training script with ResNet implementation
├── analysis/
│   └── gradient_heatmap_.png      # Gradient flow visualization
├── docs/
│   └── ResNet.pdf                 # Original paper
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── LICENSE                         # License file
```

## 🎓 Training Details

### Data Augmentation
- **Training**: 
  - Random cropping (32×32 with 4-pixel padding)
  - Random horizontal flip (p=0.5)
  - Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
- **Testing**: 
  - Center crop
  - Normalization only

### Optimization
- **Optimizer**: SGD with momentum (β=0.9)
- **Learning Rate**: 
  - Initial: 0.1
  - Schedule: Multiply by 0.1 at 32k and 48k iterations
  - Warm restarts every 205 epochs
- **Weight Decay**: 1e-4
- **Batch Size**: 128

### Initialization
- **Convolutional layers**: Kaiming initialization (He initialization)
- **Batch Normalization**: γ=1, β=0

## 📈 Monitoring

The training script includes:
- Real-time progress bars (tqdm)
- Loss and accuracy tracking
- Learning rate scheduling
- Checkpoint saving (best model + latest)
- Gradient flow analysis
- Training/validation curve plots

## 🔬 Research Notes

This implementation is designed for research and educational purposes. Key features:
- **Reproducibility**: Fixed random seeds
- **Gradient Analysis**: Visualization of gradient flow through layers
- **Ablation Studies**: Easy modification for experiments
- **Documentation**: Comprehensive code comments

## 📚 References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
2. [Original Paper (arXiv)](https://arxiv.org/abs/1512.03385)
3. [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

**Pratham Patil**
- GitHub: [@PraTham-Patill](https://github.com/PraTham-Patill)

## 🙏 Acknowledgments

- Original ResNet paper authors
- PyTorch team for the excellent framework
- CIFAR-10 dataset creators

---

⭐ If you find this implementation helpful, please consider giving it a star!
