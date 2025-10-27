import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision.models import resnet18
from datasets import FrameVideoDataset


# --- Base CNN ------------------------------------------------------
class BaseFrameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # remove final FC
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        feat = self.feature_extractor(x).flatten(1)
        logits = self.fc(feat)
        logits = logits.view(B, T, -1)
        return logits


# --- Aggregation model ------------------------------------------------------
class AggregationModel(nn.Module):
    """
    Wraps the base per-frame CNN and aggregates per-frame logits.
    """
    def __init__(self, base_model, agg='mean'):
        super().__init__()
        self.base = base_model
        self.agg = agg

    def forward(self, x):
        frame_logits = self.base(x)  # [B, T, num_classes]
        if self.agg == 'mean':
            video_logits = frame_logits.mean(dim=1)
        elif self.agg == 'max':
            video_logits = frame_logits.max(dim=1).values
        else:
            raise ValueError(f"Unsupported aggregation type: {self.agg}")
        return video_logits  # [B, num_classes]


# --- Evaluation helper ------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)  # [B, num_classes]
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 0.0 if total == 0 else correct / total


# --- Main training loop ------------------------------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    transform = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = FrameVideoDataset(root_dir='ucf101_noleakage', split='train', transform=transform, stack_frames=True)
    val_ds = FrameVideoDataset(root_dir='ucf101_noleakage', split='val', transform=transform, stack_frames=True)
    test_ds = FrameVideoDataset(root_dir='ucf101_noleakage', split='test', transform=transform, stack_frames=True)  # <-- NEW

    # Loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)  # <-- NEW

    num_classes = len(train_ds.df['label'].unique())
    print(f"Detected num_classes={num_classes}, train samples={len(train_ds)}, val samples={len(val_ds)}, test samples={len(test_ds)}")

    base = BaseFrameCNN(num_classes=num_classes)
    model = AggregationModel(base_model=base, agg='mean').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop ---
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for i, (frames, labels) in enumerate(train_loader, 1):
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)  # [B, num_classes]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if args.debug and i >= args.debug_batches:
                break

        epoch_loss = running_loss / running_total if running_total else 0.0
        epoch_acc = running_correct / running_total if running_total else 0.0
        val_acc = evaluate(model, val_loader, device=device)

        print(f"Epoch {epoch}/{args.epochs} | loss={epoch_loss:.4f} | train_acc={epoch_acc:.4f} | val_acc={val_acc:.4f}")
        if args.debug:
            break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed:.1f}s. Final val_acc={val_acc:.4f}")

    # --- Final testing phase ---
    print("\nEvaluating on test set...")
    test_acc = evaluate(model, test_loader, device=device)
    print(f"Test accuracy: {test_acc:.4f}")


# --- Entry point ------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--debug', action='store_true', help='Run short debug session (few batches)')
    parser.add_argument('--debug-batches', type=int, default=5)
    args = parser.parse_args()

    main(args)
