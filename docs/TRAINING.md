# Training Guide

This guide provides detailed information on training ResNet-56 on CIFAR-10.

## Quick Start

```bash
python src/train_resnet56_cifar10.py
```

## Hyperparameters

### Optimizer Configuration
```python
BATCH_SIZE = 128          # Mini-batch size
BASE_LR = 0.1            # Initial learning rate
MOMENTUM = 0.9           # SGD momentum
WEIGHT_DECAY = 1e-4      # L2 regularization
```

### Training Schedule
```python
TOTAL_EPOCHS = 205       # Total training epochs
LR_DECAY_EPOCHS = [102, 154]  # Learning rate decay milestones
LR_DECAY_FACTOR = 0.1    # Learning rate multiplier at milestones
```

These settings correspond to:
- 64,000 total iterations
- Decay at 32,000 and 48,000 iterations
- ~312.5 iterations per epoch (50,000 samples ÷ 128 batch size)

## Data Augmentation

### Training Set
1. **Random Crop**: 32×32 with 4-pixel padding
2. **Random Horizontal Flip**: 50% probability
3. **Normalization**: 
   - Mean: [0.4914, 0.4822, 0.4465]
   - Std: [0.2470, 0.2435, 0.2616]

### Validation/Test Set
1. **No augmentation**
2. **Normalization only**: Same as training

## Learning Rate Schedule

The learning rate follows a step decay schedule:

```
Epochs 1-102:    LR = 0.1
Epochs 103-154:  LR = 0.01
Epochs 155-205:  LR = 0.001
```

Additionally, warm restarts are applied every 205 epochs for extended training.

## Weight Initialization

- **Convolutional Layers**: Kaiming (He) initialization
  ```python
  nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
  ```
- **Batch Normalization**:
  - Weight (γ): 1.0
  - Bias (β): 0.0

## Training Pipeline

### 1. Data Loading
```python
# Training set
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# Test set
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])
```

### 2. Model Creation
```python
model = ResNet56(num_classes=10)
model = model.to(device)
```

### 3. Optimizer Setup
```python
optimizer = optim.SGD(
    model.parameters(),
    lr=BASE_LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)
```

### 4. Learning Rate Scheduler
```python
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=LR_DECAY_EPOCHS,
    gamma=LR_DECAY_FACTOR
)
```

### 5. Training Loop
```python
for epoch in range(TOTAL_EPOCHS):
    # Training phase
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            # Calculate metrics
    
    scheduler.step()
```

## Monitoring Training

### Metrics Tracked
- **Training Loss**: Cross-entropy loss on training set
- **Training Accuracy**: Top-1 accuracy on training set
- **Validation Loss**: Cross-entropy loss on test set
- **Validation Accuracy**: Top-1 accuracy on test set
- **Learning Rate**: Current learning rate

### Checkpointing
The training script saves:
- **Best Model**: Model with highest validation accuracy
- **Latest Checkpoint**: Current epoch state for resuming

### Gradient Analysis
Gradient flow is monitored to detect:
- Vanishing gradients
- Exploding gradients
- Layer-wise gradient statistics

## Expected Results

### Training Time
- **GPU (CUDA)**: ~4-6 hours
- **CPU**: ~48-72 hours (not recommended)

### Accuracy Targets
- **Training Accuracy**: ~99%+
- **Validation Accuracy**: ~93-94%
- **Test Accuracy**: ~93.03% (as reported in paper)

### Convergence Pattern
- Epochs 1-50: Rapid improvement
- Epochs 51-102: Steady improvement
- Epochs 103-154: Fine-tuning (LR=0.01)
- Epochs 155-205: Final refinement (LR=0.001)

## Troubleshooting

### Low Accuracy
- Check data augmentation
- Verify learning rate schedule
- Ensure proper weight initialization
- Check for bugs in residual connections

### Overfitting
- Increase weight decay
- Add more data augmentation
- Use dropout (not in original paper)

### Slow Training
- Reduce batch size if memory is an issue
- Use mixed precision training (FP16)
- Ensure data loading is optimized

### Gradient Issues
- Check for NaN values
- Verify batch normalization
- Monitor gradient norms

## Advanced Options

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training
For multi-GPU training, use PyTorch's DistributedDataParallel:
```python
model = nn.parallel.DistributedDataParallel(model)
```

## Reproducibility

To ensure reproducible results:
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

⚠️ **Note**: Setting `cudnn.benchmark = False` may reduce performance slightly.

## References

- He, K., et al. (2016). Deep Residual Learning for Image Recognition.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
