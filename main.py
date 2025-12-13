import torch
import torch.nn as nn
from models.srnet import SRNet
from training.trainer import SRNetTrainer

def run_sanity_check():
    print("Starting SRNet Sanity Check...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    try:
        model = SRNet(num_classes=2)
        print(f"SRNet model loaded successfully!")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Failed to load SRNet: {e}")
        return

    try:
        dummy_input = torch.randn(4, 1, 256, 256).to(device)
        dummy_labels = torch.tensor([0, 1, 0, 1]).to(device)

        trainer = SRNetTrainer(model, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("\nRunning a dummy training step...")
        loss, acc = trainer.train_step(dummy_input, dummy_labels, optimizer, criterion)

        print(f"Success! Step completed.")
        print(f"Loss: {loss:.4f}, Accuracy: {acc * 100:.1f}%")

    except Exception as e:
        print(f"Error during runtime: {e}")


if __name__ == "__main__":
    run_sanity_check()