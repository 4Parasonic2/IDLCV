import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from time import time
import numpy as np
import matplotlib.pyplot as plt

# Import your DRIVE dataset
from datarepresenattionDRIVE import DRIVEDataset, DATASET_ROOT


# ==========================================
# U-NET ARCHITECTURE
# ==========================================
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Final output
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)

# ==========================================
# METRICS IMPLEMENTATION
# ==========================================
def dice_coefficient(pred, target, eps=1e-6):
    """Dice Coefficient (F1 Score)"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    """Intersection over Union (IoU / Jaccard Index)"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

def pixel_accuracy(pred, target):
    """Pixel-wise Accuracy"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return correct / total

def sensitivity(pred, target, eps=1e-6):
    """Sensitivity (Recall / True Positive Rate)"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)

def specificity(pred, target, eps=1e-6):
    """Specificity (True Negative Rate)"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + eps) / (tn + fp + eps)

# ==========================================
# LOSS FUNCTIONS
# ==========================================

class BCELoss(nn.Module):
    """Standard Binary Cross Entropy Loss"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return self.bce(pred, target)


class WeightedBCELoss(nn.Module):
    """Binary Cross Entropy with Positive Class Weighting"""
    def __init__(self, pos_weight=7.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
    
    def forward(self, pred, target):
        bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(pred.device))
        return bce(pred, target)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


# ==========================================
# DATA LOADING AND SPLITTING
# ==========================================
size = 128
batch_size = 4

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()
])

# Load the training dataset only
full_trainset = DRIVEDataset(root=DATASET_ROOT, train=True, transform=transform)

# Split training data into train (80%) and validation (20%)
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

# Set seed for reproducibility
torch.manual_seed(42)
trainset, valset = random_split(full_trainset, [train_size, val_size])

# Load test dataset
testset = DRIVEDataset(root=DATASET_ROOT, train=False, transform=transform)

if len(trainset) == 0 or len(valset) == 0:
    raise RuntimeError(f"No data found in {DATASET_ROOT}.")

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

print("="*60)
print("DRIVE DATASET - LOSS FUNCTION COMPARISON")
print("="*60)
print(f"Training samples:   {len(trainset)}")
print(f"Validation samples: {len(valset)}")
print(f"Test samples:       {len(testset)}")
print(f"Batch size: {batch_size}")
print(f"Image size: {size}x{size}")
print("="*60)

# ==========================================
# DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ==========================================
# TRAINING FUNCTION
# ==========================================
def train_model(loss_name, loss_fn, epochs=10):
    """Train model with specified loss function"""
    print(f"\n{'='*60}")
    print(f"Training with: {loss_name}")
    print(f"{'='*60}")
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_acc': [],
        'val_sens': [],
        'val_spec': []
    }
    
    best_dice = 0.0
    
    for epoch in range(epochs):
        tic = time()
        
        # TRAINING
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_acc = 0.0
        val_sens = 0.0
        val_spec = 0.0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                
                loss = loss_fn(y_pred, y_val)
                val_loss += loss.item()
                
                val_dice += dice_coefficient(y_pred, y_val).item()
                val_iou += iou_score(y_pred, y_val).item()
                val_acc += pixel_accuracy(y_pred, y_val).item()
                val_sens += sensitivity(y_pred, y_val).item()
                val_spec += specificity(y_pred, y_val).item()
        
        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        avg_val_dice = val_dice / n_val
        avg_val_iou = val_iou / n_val
        avg_val_acc = val_acc / n_val
        avg_val_sens = val_sens / n_val
        avg_val_spec = val_spec / n_val
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)
        history['val_iou'].append(avg_val_iou)
        history['val_acc'].append(avg_val_acc)
        history['val_sens'].append(avg_val_sens)
        history['val_spec'].append(avg_val_spec)
        
        elapsed = time() - tic
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Dice: {avg_val_dice:.4f} | "
              f"IoU: {avg_val_iou:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
    
    # Final test evaluation
    model.eval()
    test_metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'acc': 0.0,
        'sens': 0.0,
        'spec': 0.0
    }
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            
            test_metrics['dice'] += dice_coefficient(y_pred, y_test).item()
            test_metrics['iou'] += iou_score(y_pred, y_test).item()
            test_metrics['acc'] += pixel_accuracy(y_pred, y_test).item()
            test_metrics['sens'] += sensitivity(y_pred, y_test).item()
            test_metrics['spec'] += specificity(y_pred, y_test).item()
    
    n_test = len(test_loader)
    for key in test_metrics:
        test_metrics[key] /= n_test
    
    print(f"\nFinal Test Results for {loss_name}:")
    print(f"  Dice: {test_metrics['dice']:.4f}")
    print(f"  IoU:  {test_metrics['iou']:.4f}")
    print(f"  Acc:  {test_metrics['acc']:.4f}")
    print(f"  Sens: {test_metrics['sens']:.4f}")
    print(f"  Spec: {test_metrics['spec']:.4f}")
    
    return history, test_metrics

# ==========================================
# TRAIN WITH EACH LOSS FUNCTION
# ==========================================

loss_functions = {
    'Cross Entropy': BCELoss(),
    'Weighted Cross Entropy': WeightedBCELoss(pos_weight=7.0),
    'Focal Loss': FocalLoss(alpha=0.25, gamma=2.0)
}

results = {}
epochs = 10

for loss_name, loss_fn in loss_functions.items():
    history, test_metrics = train_model(loss_name, loss_fn, epochs=epochs)
    results[loss_name] = {
        'history': history,
        'test_metrics': test_metrics
    }

# ==========================================
# PLOT RESULTS
# ==========================================
print(f"\n{'='*60}")
print("GENERATING COMPARISON PLOTS")
print(f"{'='*60}")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Loss Function Comparison - DRIVE Dataset U-Net', fontsize=16, fontweight='bold')

colors = {'Cross Entropy': '#3498db', 
          'Weighted Cross Entropy': '#e74c3c', 
          'Focal Loss': '#2ecc71'}

epochs_range = range(1, epochs + 1)

# Plot 1: Training Loss
ax = axes[0, 0]
for loss_name in loss_functions.keys():
    ax.plot(epochs_range, results[loss_name]['history']['train_loss'], 
            label=loss_name, color=colors[loss_name], linewidth=2, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Validation Loss
ax = axes[0, 1]
for loss_name in loss_functions.keys():
    ax.plot(epochs_range, results[loss_name]['history']['val_loss'], 
            label=loss_name, color=colors[loss_name], linewidth=2, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Validation Loss')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Dice Coefficient
ax = axes[0, 2]
for loss_name in loss_functions.keys():
    ax.plot(epochs_range, results[loss_name]['history']['val_dice'], 
            label=loss_name, color=colors[loss_name], linewidth=2, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Dice Score')
ax.set_title('Validation Dice Coefficient')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: IoU Score
ax = axes[1, 0]
for loss_name in loss_functions.keys():
    ax.plot(epochs_range, results[loss_name]['history']['val_iou'], 
            label=loss_name, color=colors[loss_name], linewidth=2, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('IoU Score')
ax.set_title('Validation IoU (Jaccard Index)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Accuracy
ax = axes[1, 1]
for loss_name in loss_functions.keys():
    ax.plot(epochs_range, results[loss_name]['history']['val_acc'], 
            label=loss_name, color=colors[loss_name], linewidth=2, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Validation Accuracy')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Sensitivity
ax = axes[1, 2]
for loss_name in loss_functions.keys():
    ax.plot(epochs_range, results[loss_name]['history']['val_sens'], 
            label=loss_name, color=colors[loss_name], linewidth=2, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Sensitivity')
ax.set_title('Validation Sensitivity (Recall)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 7: Specificity
ax = axes[2, 0]
for loss_name in loss_functions.keys():
    ax.plot(epochs_range, results[loss_name]['history']['val_spec'], 
            label=loss_name, color=colors[loss_name], linewidth=2, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Specificity')
ax.set_title('Validation Specificity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 8: Test Set Metrics Comparison (Bar Chart)
ax = axes[2, 1]
metrics = ['dice', 'iou', 'acc', 'sens', 'spec']
metric_labels = ['Dice', 'IoU', 'Accuracy', 'Sensitivity', 'Specificity']
x = np.arange(len(metrics))
width = 0.25

for i, loss_name in enumerate(loss_functions.keys()):
    values = [results[loss_name]['test_metrics'][m] for m in metrics]
    ax.bar(x + i*width, values, width, label=loss_name, color=colors[loss_name])

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Final Test Set Performance')
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 9: Summary Table
ax = axes[2, 2]
ax.axis('off')
table_data = []
for loss_name in loss_functions.keys():
    row = [loss_name,
           f"{results[loss_name]['test_metrics']['dice']:.3f}",
           f"{results[loss_name]['test_metrics']['iou']:.3f}",
           f"{results[loss_name]['test_metrics']['sens']:.3f}",
           f"{results[loss_name]['test_metrics']['spec']:.3f}"]
    table_data.append(row)

table = ax.table(cellText=table_data,
                colLabels=['Loss Function', 'Dice', 'IoU', 'Sens', 'Spec'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
for i, loss_name in enumerate(loss_functions.keys()):
    table[(i+1, 0)].set_facecolor(colors[loss_name])
    table[(i+1, 0)].set_text_props(color='white', weight='bold')

ax.set_title('Test Set Summary', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('loss_function_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot to 'loss_function_comparison.png'")
plt.show()

print(f"\n{'='*60}")
print("✅ ALL EXPERIMENTS COMPLETED!")
print(f"{'='*60}")