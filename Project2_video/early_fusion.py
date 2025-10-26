import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision.models import resnet18, resnet50
from datasets import FrameVideoDataset


class VideoEarlyFusionResNet(nn.Module):
    def __init__(self, num_classes, num_frames=10, input_channels=3):
        super(VideoEarlyFusionResNet, self).__init__()
        
        # Total input channels after concatenating all frames
        fused_channels = input_channels * num_frames
        
        # Load pre-trained ResNet-18
        self.resnet = resnet18(pretrained=True)
        
        # Modify first conv layer to accept fused input: C * F channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels=fused_channels,
            out_channels=64,
            kernel_size=3,  # ResNet-18 default
            stride=1,
            padding=1,
            bias=False
        )
        
        # Modify maxpool if needed (usually keep same)
        # Optional: adjust stride/padding if spatial downsampling is too aggressive
        
        # Replace final FC layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        self.num_frames = num_frames
        self.input_channels = input_channels

    def forward(self, x):
        # Expected input shape: [batch, channels * frames, H, W]
        # e.g., [B, 3*16, 64, 64] = [B, 48, 64, 64]
        
        batch_size = x.size(0)
        
        # Ensure input has correct number of channels
        expected_channels = self.input_channels * self.num_frames
        assert x.size(1) == expected_channels, \
            f"Expected {expected_channels} channels (C*F), got {x.size(1)}"
        
        # Pass through modified ResNet directly
        out = self.resnet(x)  # [B, num_classes]
        
        return out

# --- Evaluation helper ------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)

            fused_frames = frames.view(frames.size(0), -1, frames.size(3), frames.size(4))
            outputs = model(fused_frames)  # [B, num_classes]
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 0.0 if total == 0 else correct / total


# --- Main training loop -----------------------------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # optional
    ])

    train_ds = FrameVideoDataset(root_dir='/dtu/datasets1/02516/ufc10/', split='train', transform=transform, stack_frames=True)
    val_ds = FrameVideoDataset(root_dir='/dtu/datasets1/02516/ufc10/', split='val', transform=transform, stack_frames=True)
    test_ds = FrameVideoDataset(root_dir='/dtu/datasets1/02516/ufc10/', split='test', transform=transform, stack_frames=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_ds.df['label'].unique())
    print(f"Detected num_classes={num_classes}, train samples={len(train_ds)}, val samples={len(val_ds)}")

    model = VideoEarlyFusionResNet(num_classes=num_classes, num_frames=10, input_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_epoch = 0
    best_loss = 10**3
    best_val_acc = 0
    best_model_state = model.state_dict()

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for i, (frames, labels) in enumerate(train_loader, 1):
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            fused_frames = frames.view(frames.size(0), -1, frames.size(3), frames.size(4))
            # print("fused: ", fused_frames.shape)
            outputs = model(fused_frames)  # [B, num_classes]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch

            if args.debug and i >= args.debug_batches:
                break

        epoch_loss = running_loss / running_total if running_total else 0.0
        epoch_acc = running_correct / running_total if running_total else 0.0
        val_acc = evaluate(model, val_loader, device=device)

        if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()

        # print(f"Epoch {epoch}/{args.epochs} | loss={epoch_loss:.4f} | train_acc={epoch_acc:.4f} | val_acc={val_acc:.4f}")
        if args.debug:
            break

    elapsed = time.time() - start


    # Load best model
    model.load_state_dict(best_model_state)

    test_acc = evaluate(model, test_loader, device=device)
    print(f"Finished in {elapsed:.1f}s. Final val_acc={val_acc:.4f}")

    print(f"Best loss: {best_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc}")
    print(f"Test accuracy: {test_acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--debug', action='store_true', help='Run short debug session (few batches)')
    parser.add_argument('--debug-batches', type=int, default=5)
    args = parser.parse_args()

    main(args)