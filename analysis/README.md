# Analysis

This folder contains analysis outputs and visualizations from training ResNet-56 on CIFAR-10.

## Contents

### Gradient Flow Analysis
- **gradient_heatmap_.png**: Visualization of gradient magnitudes across layers
  - Shows gradient flow through the network
  - Helps identify potential vanishing/exploding gradient issues
  - Demonstrates effectiveness of skip connections

## How to Generate Analysis

The training script automatically generates analysis outputs during training:

```bash
python src/train_resnet56_cifar10.py
```

## Interpreting Results

### Gradient Heatmap
- **Purpose**: Visualize how gradients flow backward through the network
- **What to look for**:
  - Uniform gradient magnitudes across layers indicate healthy training
  - Very small gradients (vanishing) indicate learning difficulty
  - Very large gradients (exploding) indicate instability
  - Skip connections should help maintain gradient flow

### Expected Observations
With ResNet's skip connections, you should see:
- Consistent gradient magnitudes throughout the network
- No significant vanishing in early layers
- Stable gradient flow even in deep layers

## Additional Analysis (Future Work)

Future analysis could include:
- Training/validation loss curves
- Accuracy progression over epochs
- Learning rate vs. accuracy plots
- Per-class accuracy confusion matrices
- Feature map visualizations
- Activation distributions
- Weight distribution evolution

## Tools Used

- **PyTorch**: For gradient computation
- **Matplotlib**: For visualization
- **NumPy**: For numerical analysis
