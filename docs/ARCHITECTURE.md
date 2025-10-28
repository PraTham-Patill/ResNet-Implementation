# ResNet-56 Architecture Details

## Overview
This document provides detailed information about the ResNet-56 architecture implementation for CIFAR-10.

## Network Architecture

### Input Layer
- **Input Size**: 32×32×3 (CIFAR-10 images)
- **Preprocessing**: 
  - Normalization with CIFAR-10 statistics
  - Mean: [0.4914, 0.4822, 0.4465]
  - Std: [0.2470, 0.2435, 0.2616]

### Initial Convolution
```
Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
BatchNorm2d(16)
ReLU()
```

### Residual Blocks

The network consists of 3 stages with 9 residual blocks each:

#### Stage 1: 9 blocks
- **Input/Output Channels**: 16
- **Feature Map Size**: 32×32
- **Stride**: 1
- **Blocks**: 9 × BasicBlock(16, 16, stride=1)

#### Stage 2: 9 blocks
- **Input Channels**: 16
- **Output Channels**: 32
- **Feature Map Size**: 16×16
- **First Block Stride**: 2 (downsampling)
- **Remaining Blocks Stride**: 1
- **Blocks**: BasicBlock(16, 32, stride=2) + 8 × BasicBlock(32, 32, stride=1)

#### Stage 3: 9 blocks
- **Input Channels**: 32
- **Output Channels**: 64
- **Feature Map Size**: 8×8
- **First Block Stride**: 2 (downsampling)
- **Remaining Blocks Stride**: 1
- **Blocks**: BasicBlock(32, 64, stride=2) + 8 × BasicBlock(64, 64, stride=1)

### Global Average Pooling
```
AdaptiveAvgPool2d((1, 1))
```
- Reduces 8×8 feature maps to 1×1

### Fully Connected Layer
```
Linear(64, 10)
```
- Maps 64 features to 10 CIFAR-10 classes

## Basic Block Structure

Each BasicBlock contains:
```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # First 3×3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second 3×3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (projection if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out
```

## Layer Count Calculation

Total layers in ResNet-56:
- 1 initial convolution layer
- 27 residual blocks × 2 convolutions per block = 54 layers
- 1 fully connected layer

**Total**: 1 + 54 + 1 = **56 layers**

## Parameter Count

Approximate parameter breakdown:
- **Initial Conv**: 3×3×3×16 = 432
- **Stage 1 (9 blocks)**: ~73K parameters
- **Stage 2 (9 blocks)**: ~294K parameters
- **Stage 3 (9 blocks)**: ~1.2M parameters
- **FC Layer**: 64×10 = 640
- **Batch Norm parameters**: ~10K

**Total Parameters**: ~855,770 parameters

## Key Design Principles

1. **Skip Connections**: Enable gradient flow through very deep networks
2. **Batch Normalization**: After every convolution, before activation
3. **No Bias Terms**: In convolutional layers (BN handles bias)
4. **Projection Shortcuts**: Used when dimensions change (stride=2 or channel mismatch)
5. **Global Average Pooling**: Reduces parameters and improves generalization

## Comparison with Other ResNet Variants

| Model | Layers | Parameters | CIFAR-10 Accuracy |
|-------|--------|------------|-------------------|
| ResNet-20 | 20 | ~270K | ~91.25% |
| ResNet-32 | 32 | ~470K | ~92.49% |
| ResNet-44 | 44 | ~660K | ~92.83% |
| **ResNet-56** | **56** | **~855K** | **~93.03%** |
| ResNet-110 | 110 | ~1.7M | ~93.57% |
| ResNet-1202 | 1202 | ~19.4M | ~92.07% (overfitting) |

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. ECCV 2016.
