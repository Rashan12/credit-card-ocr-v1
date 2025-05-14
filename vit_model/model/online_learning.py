import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as transforms
from api.metrics_db import MetricsDB
import os

class OnlineLearner:
    def __init__(self, model, learning_rate=5e-3, T_max=150, eta_min=1e-6, weight_decay=1e-4):
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        self.criterion = CrossEntropyLoss()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.metrics_db = MetricsDB()
        self.step_count = 0

    def train_step(self, image, card_number_target, expiry_target, security_target):
        self.model.train()
        self.optimizer.zero_grad()

        # Transform the image to a tensor first
        image_tensor = self.transform(image)
        
        # Now that image_tensor is a PyTorch tensor, we can safely check its dimensions
        image_tensor = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
        card_logits, expiry_logits, security_logits = self.model(image_tensor)

        card_target = torch.tensor([card_number_target], dtype=torch.long)
        expiry_target = torch.tensor([expiry_target], dtype=torch.long)
        security_target = torch.tensor([security_target], dtype=torch.long)

        card_loss = self.criterion(card_logits.view(-1, 10), card_target.view(-1))
        expiry_loss = self.criterion(expiry_logits.view(-1, 10), expiry_target.view(-1))
        security_loss = self.criterion(security_logits.view(-1, 10), security_target.view(-1))
        total_loss = card_loss + expiry_loss + security_loss

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        total_loss.backward()
        self.optimizer.step()

        # Early stopping if loss is below a threshold (e.g., 0.01)
        if total_loss.item() < 0.01:
            return total_loss.item(), card_loss.item(), expiry_loss.item(), security_loss.item(), True
        return total_loss.item(), card_loss.item(), expiry_loss.item(), security_loss.item(), False

    def update_with_feedback(self, image, feedback):
        card_number = feedback['card_number'].replace(" ", "")
        card_target = [int(d) for d in card_number.ljust(16, '0')[:16]]

        expiry = feedback['expiry_date'].replace("/", "")
        expiry_target = [int(d) for d in expiry.ljust(4, '0')[:4]]

        security = feedback.get('cvv', '').ljust(7, '0')[:7]
        security_target = [int(d) for d in security]

        for epoch in range(5):  # Increased from 3 to 5 for more learning
            total_loss, card_loss, expiry_loss, security_loss, stop_early = self.train_step(
                image, card_target, expiry_target, security_target
            )
            self.step_count += 1
            learning_rate = self.optimizer.param_groups[0]['lr']
            print(f"Online learning step {self.step_count}, Epoch {epoch+1}/5. "
                  f"Total Loss: {total_loss}, Card Loss: {card_loss}, Expiry Loss: {expiry_loss}, "
                  f"Security Loss: {security_loss}, Learning Rate: {learning_rate}")
            self.metrics_db.log_training(self.step_count, epoch + 1, total_loss, card_loss, expiry_loss, security_loss, learning_rate)
            if stop_early:
                break
        
        self.scheduler.step()
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "api", "vit_model_weights.pth")
        self.model.save(save_path)