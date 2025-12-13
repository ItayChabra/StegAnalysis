import torch

class SRNetTrainer:
    """Utility class for training SRNet"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def train_step(self, images, labels, optimizer, criterion):
        """Single training step"""
        self.model.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)

        return loss.item(), accuracy

    def validate(self, val_loader, criterion):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy