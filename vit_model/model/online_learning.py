import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as transforms
from api.metrics_db import MetricsDB
import os
import numpy as np

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
        self.feedback_buffer = []  # Buffer to store feedback for batch learning

    def train_step(self, image, card_number_target, expiry_target, security_target):
        self.model.train()
        self.optimizer.zero_grad()

        image_tensor = self.transform(image)
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

        if total_loss.item() < 0.01:
            return total_loss.item(), card_loss.item(), expiry_loss.item(), security_loss.item(), True
        return total_loss.item(), card_loss.item(), expiry_loss.item(), security_loss.item(), False

    def process_feedback_batch(self):
        if not self.feedback_buffer:
            return

        images = []
        card_targets = []
        expiry_targets = []
        security_targets = []

        for feedback_item in self.feedback_buffer:
            image, feedback = feedback_item
            card_number = feedback['card_number'].replace(" ", "")
            card_target = [int(d) for d in card_number.ljust(16, '0')[:16]]
            expiry = feedback['expiry_date'].replace("/", "")
            expiry_target = [int(d) for d in expiry.ljust(4, '0')[:4]]
            security = feedback.get('cvv', '').ljust(7, '0')[:7]
            security_target = [int(d) for d in security]

            images.append(self.transform(image).unsqueeze(0))
            card_targets.append(card_target)
            expiry_targets.append(expiry_target)
            security_targets.append(security_target)

        images_tensor = torch.cat(images, dim=0).to(next(self.model.parameters()).device)
        card_targets_tensor = torch.tensor(card_targets, dtype=torch.long).to(next(self.model.parameters()).device)
        expiry_targets_tensor = torch.tensor(expiry_targets, dtype=torch.long).to(next(self.model.parameters()).device)
        security_targets_tensor = torch.tensor(security_targets, dtype=torch.long).to(next(self.model.parameters()).device)

        self.model.train()
        self.optimizer.zero_grad()
        card_logits, expiry_logits, security_logits = self.model(images_tensor)

        card_loss = self.criterion(card_logits.view(-1, 10), card_targets_tensor.view(-1))
        expiry_loss = self.criterion(expiry_logits.view(-1, 10), expiry_targets_tensor.view(-1))
        security_loss = self.criterion(security_logits.view(-1, 10), security_targets_tensor.view(-1))
        total_loss = card_loss + expiry_loss + security_loss

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        total_loss.backward()
        self.optimizer.step()

        self.step_count += len(self.feedback_buffer)
        learning_rate = self.optimizer.param_groups[0]['lr']
        print(f"Online learning step {self.step_count}, Batch Size: {len(self.feedback_buffer)}. "
              f"Total Loss: {total_loss.item()}, Card Loss: {card_loss.item()}, "
              f"Expiry Loss: {expiry_loss.item()}, Security Loss: {security_loss.item()}, "
              f"Learning Rate: {learning_rate}")
        self.metrics_db.log_training(self.step_count, 1, total_loss.item(), card_loss.item(), expiry_loss.item(), security_loss.item(), learning_rate)

        if total_loss.item() < 0.01:
            self.feedback_buffer.clear()
        else:
            self.feedback_buffer = self.feedback_buffer[-5:]  # Keep last 5 for next batch if needed

        self.scheduler.step()
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "api", "vit_model_weights.pth")
        self.model.save(save_path)

    def update_with_feedback(self, image, feedback, num_epochs=5):
        card_number = feedback['card_number'].replace(" ", "")
        card_target = [int(d) for d in card_number.ljust(16, '0')[:16]]

        expiry = feedback['expiry_date'].replace("/", "")
        expiry_target = [int(d) for d in expiry.ljust(4, '0')[:4]]

        security = feedback.get('cvv', '').ljust(7, '0')[:7]
        security_target = [int(d) for d in security]

        self.feedback_buffer.append((image, feedback))

        # Process batch if buffer size reaches 5 or on proceed
        if len(self.feedback_buffer) >= 5:
            self.process_feedback_batch()

        # Immediate learning for single feedback if no batching yet
        if len(self.feedback_buffer) == 1:
            for epoch in range(num_epochs):
                total_loss, card_loss, expiry_loss, security_loss, stop_early = self.train_step(
                    image, card_target, expiry_target, security_target
                )
                self.step_count += 1
                learning_rate = self.optimizer.param_groups[0]['lr']
                print(f"Online learning step {self.step_count}, Epoch {epoch+1}/{num_epochs}. "
                      f"Total Loss: {total_loss}, Card Loss: {card_loss}, Expiry Loss: {expiry_loss}, "
                      f"Security Loss: {security_loss}, Learning Rate: {learning_rate}")
                self.metrics_db.log_training(self.step_count, epoch + 1, total_loss, card_loss, expiry_loss, security_loss, learning_rate)
                if stop_early:
                    break

            self.scheduler.step()
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "api", "vit_model_weights.pth")
            self.model.save(save_path)