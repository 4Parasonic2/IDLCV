import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision.models import resnet18
from datasets import FrameVideoDataset, FlowVideoDataset  # <-- you need to implement FlowVideoDataset
from datasets import FlowTransform
import torch.nn.functional as F


# --- Spatial Stream CNN (RGB) ------------------------------------------------
class BaseFrameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        feat = self.feature_extractor(x).flatten(1)
        logits = self.fc(feat)
        logits = logits.view(B, T, -1)
        return logits


# --- Temporal Stream CNN (Optical Flow) -------------------------------------
class FlowFrameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=None)  # flow is not RGB; start from scratch or use finetuned weights
        # modify first conv to accept 2-channel input (u, v)
        base.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape  # C=2 for flow
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        feat = self.feature_extractor(x).flatten(1)
        logits = self.fc(feat)
        logits = logits.view(B, T, -1)
        return logits


# --- Aggregation Model (mean pooling) ---------------------------------------
class AggregationModel(nn.Module):
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

class DualStreamDataset(torch.utils.data.Dataset):
    """
    Returns a tuple of (RGB frames, Flow frames) and the label.
    Assumes both datasets are aligned and have the same order of videos.
    """
    def __init__(self, rgb_root, flow_root, split='train',
                 rgb_transform=None, flow_transform=None, stack_frames=True):
        self.rgb_ds = FrameVideoDataset(root_dir=rgb_root, split=split,
                                        transform=rgb_transform, stack_frames=stack_frames)
        self.flow_ds = FlowVideoDataset(root_dir=flow_root, split=split,
                                        transform=flow_transform, stack_frames=stack_frames)
        # Quick sanity check
        assert len(self.rgb_ds) == len(self.flow_ds), \
            f"RGB and Flow dataset lengths mismatch: {len(self.rgb_ds)} vs {len(self.flow_ds)}"

        # Store metadata for convenience
        self.df = self.rgb_ds.df

    def __len__(self):
        return len(self.rgb_ds)

    def __getitem__(self, idx):
        rgb_frames, label_rgb = self.rgb_ds[idx]
        flow_frames, label_flow = self.flow_ds[idx]

        if label_rgb != label_flow:
            raise ValueError(f"Label mismatch at idx {idx}: RGB={label_rgb}, Flow={label_flow}")

        return (rgb_frames, flow_frames), label_rgb


# --- Dual Stream Fusion Model -----------------------------------------------
class DualStreamModel(nn.Module):
    def __init__(self, spatial_model, temporal_model, fusion='avg'):
        super().__init__()
        self.spatial = spatial_model
        self.temporal = temporal_model
        self.fusion = fusion

    def forward(self, rgb_frames, flow_frames):
        rgb_logits = self.spatial(rgb_frames)  # [B, num_classes]
        flow_logits = self.temporal(flow_frames)
        if self.fusion == 'avg':
            return (rgb_logits + flow_logits) / 2
        elif self.fusion == 'concat':
            return torch.cat((rgb_logits, flow_logits), dim=1)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")


# --- Evaluation --------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (rgb_frames, flow_frames), labels in loader:
            rgb_frames, flow_frames, labels = rgb_frames.to(device), flow_frames.to(device), labels.to(device)
            outputs = model(rgb_frames, flow_frames)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 0.0 if total == 0 else correct / total


# --- Main training loop -----------------------------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_rgb = T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_flow = FlowTransform(size=(128,128))

    train_ds = DualStreamDataset(
        rgb_root='ucf101_noleakage', flow_root='ucf101_noleakage',
        split='train', rgb_transform=transform_rgb, flow_transform=transform_flow
    )
    val_ds = DualStreamDataset(
        rgb_root='ucf101_noleakage', flow_root='ucf101_noleakage',
        split='val', rgb_transform=transform_rgb, flow_transform=transform_flow
    )


    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_ds.df['label'].unique())
    print(f"Detected num_classes={num_classes}")

    # Build models
    spatial_model = AggregationModel(BaseFrameCNN(num_classes=num_classes))
    temporal_model = AggregationModel(FlowFrameCNN(num_classes=num_classes))
    model = DualStreamModel(spatial_model, temporal_model, fusion='avg').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for (rgb_frames, flow_frames), labels in train_loader:
            rgb_frames, flow_frames, labels = rgb_frames.to(device), flow_frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb_frames, flow_frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if args.debug and running_total >= args.debug_batches * args.batch_size:
                break

        epoch_loss = running_loss / running_total if running_total else 0.0
        epoch_acc = running_correct / running_total if running_total else 0.0
        val_acc = evaluate(model, val_loader, device=device)

        print(f"Epoch {epoch}/{args.epochs} | loss={epoch_loss:.4f} | train_acc={epoch_acc:.4f} | val_acc={val_acc:.4f}")
        if args.debug:
            break

    elapsed = time.time() - start
    print(f"Finished in {elapsed:.1f}s. Final val_acc={val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug-batches', type=int, default=5)
    args = parser.parse_args()

    main(args)
