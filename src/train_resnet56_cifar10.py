import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from tqdm import tqdm
from collections import defaultdict
import os

# Set random seeds for reproducibility
def set_seed(seed=42):
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters following the paper
BATCH_SIZE = 128
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TOTAL_EPOCHS = 205  # 64,000 iterations / 312.5 iterations_per_epoch ≈ 205 epochs
LR_DECAY_EPOCHS = [102, 154]  # 32,000 and 48,000 iterations converted to epochs
LR_DECAY_FACTOR = 0.1

# CIFAR-10 normalization statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

class BasicBlock(nn.Module):
  
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Option A shortcut: zero-padding for channel dimension increase
        self.shortcut_downsample = stride != 1
        self.channel_increase = out_channels > in_channels
        
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Option A shortcut connection
        if self.shortcut_downsample:
            # Downsample identity using average pooling
            identity = F.avg_pool2d(identity, kernel_size=1, stride=2)
        
        if self.channel_increase:
            # Zero-pad channels to match output dimensions (Option A from paper)
            batch_size, channels, height, width = identity.shape
            extra_channels = out.shape[1] - channels
            # Pad on channel dimension: (left, right, top, bottom, front, back)
            identity = F.pad(identity, (0, 0, 0, 0, 0, extra_channels))
        
        # Add shortcut and apply ReLU
        out += identity
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
   
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        
        # Initial conv layer (3x3, no 7x7 + maxpool for CIFAR-10)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Three stages with different feature map sizes and channels
        self.layer1 = self._make_layer(block, 16, 16, layers[0], stride=1)   # 32x32, 16 channels
        self.layer2 = self._make_layer(block, 16, 32, layers[1], stride=2)   # 16x16, 32 channels  
        self.layer3 = self._make_layer(block, 32, 64, layers[2], stride=2)   # 8x8, 64 channels
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights using He/Kaiming normal initialization
        self._initialize_weights()
        
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        
        layers = []
        
        # First block may have stride > 1 for downsampling
        layers.append(block(in_channels, out_channels, stride))
        
        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Three stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet56(**kwargs):
   
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)

def get_cifar10_data():
   
    # Data augmentation for training (pad + random crop + horizontal flip)
    train_transform = transforms.Compose([
        transforms.Pad(4),  # Pad 4 pixels on each side (32x32 -> 40x40)
        transforms.RandomCrop(32),  # Random crop back to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    
    # No augmentation for validation and test
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    
    # Load full training set for splitting
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # Load test set
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_test_transform
    )
    
    # Split training set into train (80%) and validation (20%)
    train_size = int(0.8 * len(full_train_dataset))  # 40,000 images
    val_size = len(full_train_dataset) - train_size   # 10,000 images
    
    # Create indices for splitting
    indices = list(range(len(full_train_dataset)))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train subset with augmentation
    train_dataset = Subset(full_train_dataset, train_indices)
    
    # Create validation dataset without augmentation
    val_full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=val_test_transform
    )
    val_dataset = Subset(val_full_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Iterations per epoch: {len(train_loader)}")
    
    return train_loader, val_loader, test_loader

def calculate_accuracy(outputs, targets):
    
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total

def compute_gradient_norms(model):
   
    grad_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Extract layer name (e.g., 'layer1.0.conv1' -> 'layer1.0')
            layer_name = '.'.join(name.split('.')[:-1])
            if layer_name not in grad_norms:
                grad_norms[layer_name] = []
            
            # Compute L2 norm of gradients for this parameter
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[layer_name].append(grad_norm)
    
    # Average gradient norms within each layer
    avg_grad_norms = {}
    for layer_name, norms in grad_norms.items():
        avg_grad_norms[layer_name] = np.mean(norms)
    
    return avg_grad_norms

def train_epoch(model, train_loader, optimizer, criterion, device):
  
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        running_acc += calculate_accuracy(outputs, targets)
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        avg_acc = running_acc / (batch_idx + 1)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.2f}%'})
    
    return running_loss / num_batches, running_acc / num_batches

def validate_epoch(model, val_loader, criterion, device):
    
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, targets)
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = running_acc / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.2f}%'})
    
    return running_loss / num_batches, running_acc / num_batches

def test_model(model, test_loader, device):
    
    model.eval()
    running_acc = 0.0
    num_batches = len(test_loader)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            running_acc += calculate_accuracy(outputs, targets)
            
            avg_acc = running_acc / (batch_idx + 1)
            pbar.set_postfix({'Acc': f'{avg_acc:.2f}%'})
    
    return running_acc / num_batches

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves saved to training_curves.png")

def plot_gradient_flow(gradient_history):
    
    if not gradient_history:
        return
    
    # Get layer names and epochs
    layer_names = list(next(iter(gradient_history.values())).keys())
    epochs = sorted(gradient_history.keys())
    
    # Create color map for different epochs
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    
    plt.figure(figsize=(15, 8))
    
    for i, epoch in enumerate(epochs):
        grad_norms = [gradient_history[epoch][layer] for layer in layer_names]
        plt.plot(range(len(layer_names)), grad_norms, 
                color=colors[i], marker='o', markersize=4, 
                label=f'Epoch {epoch}')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Average Gradient L2 Norm')
    plt.title('Gradient Flow Evolution During Training')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gradient flow plot saved to gradient_flow.png")

def save_run_summary(test_acc, total_time, hyperparameters):

    with open('run_summary.txt', 'w') as f:
        f.write("ResNet-56 CIFAR-10 Training Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n\n")
        
        f.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nTotal Training Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n")
        f.write(f"Random Seed: 42\n")
        f.write(f"Device: {device}\n")
        
        f.write("\nModel Architecture:\n")
        f.write("  ResNet-56 with Option A shortcuts\n")
        f.write("  Basic blocks: 9 + 9 + 9 = 27 blocks\n")
        f.write("  Layers: 1 initial conv + 54 conv + 1 fc = 56 layers\n")
        f.write("  Channels: 16 -> 32 -> 64\n")
        f.write("  Feature maps: 32x32 -> 16x16 -> 8x8\n")
        
    print("Training summary saved to run_summary.txt")

def main():
    
    print("Initializing ResNet-56 training on CIFAR-10")
    print("Following 'Deep Residual Learning for Image Recognition' methodology")
    print()
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_data()
    
    # Create model
    print("\nInitializing ResNet-56 model...")
    model = resnet56(num_classes=10)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR, 
                         momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler (step at epochs 102 and 154)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                              milestones=LR_DECAY_EPOCHS, 
                                              gamma=LR_DECAY_FACTOR)
    
    # Tracking variables
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    gradient_history = {}
    
    # Hyperparameters for summary
    hyperparameters = {
        'Batch Size': BATCH_SIZE,
        'Base Learning Rate': BASE_LR,
        'Momentum': MOMENTUM,
        'Weight Decay': WEIGHT_DECAY,
        'Total Epochs': TOTAL_EPOCHS,
        'LR Decay Epochs': LR_DECAY_EPOCHS,
        'LR Decay Factor': LR_DECAY_FACTOR
    }
    
    print(f"\nStarting training for {TOTAL_EPOCHS} epochs...")
    print(f"LR schedule: {BASE_LR} -> {BASE_LR * LR_DECAY_FACTOR} at epoch {LR_DECAY_EPOCHS[0]} -> {BASE_LR * LR_DECAY_FACTOR**2} at epoch {LR_DECAY_EPOCHS[1]}")
    print()
    
    start_time = time.time()
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{TOTAL_EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Compute gradient norms every 10 epochs
        if epoch % 10 == 0:
            # Compute gradients on a small batch
            model.train()
            data, targets = next(iter(train_loader))
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            gradient_history[epoch] = compute_gradient_norms(model)
            print(f"Gradient norms computed for epoch {epoch}")
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    total_time = time.time() - start_time
    
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_acc = test_model(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'resnet56_cifar10.pth')
    print("Model saved to resnet56_cifar10.pth")
    
    # Create plots
    print("\nGenerating plots...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_gradient_flow(gradient_history)
    
    # Save summary
    save_run_summary(test_acc, total_time, hyperparameters)
    
    print("\nTraining complete! All files saved:")
    print("- resnet56_cifar10.pth (model weights)")
    print("- training_curves.png (loss and accuracy plots)")
    print("- gradient_flow.png (gradient evolution)")
    print("- run_summary.txt (training summary)")

if __name__ == "__main__":
    main()