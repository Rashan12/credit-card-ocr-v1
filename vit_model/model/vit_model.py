import torch
from transformers import ViTModel, ViTConfig
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

class CreditCardViT(nn.Module):
    def __init__(self, num_classes_card=16, num_classes_expiry=4, num_classes_security=7):
        super(CreditCardViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        for layer in self.vit.encoder.layer[-6:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.card_number_head = nn.Linear(self.vit.config.hidden_size, num_classes_card * 10)
        self.expiry_head = nn.Linear(self.vit.config.hidden_size, num_classes_expiry * 10)
        self.security_head = nn.Linear(self.vit.config.hidden_size, num_classes_security * 10)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def forward(self, images):
        if isinstance(images, Image.Image):
            images = self.transform(images).unsqueeze(0)
        
        outputs = self.vit(pixel_values=images)
        cls_output = outputs.last_hidden_state[:, 0, :]

        card_number_logits = self.card_number_head(cls_output)
        expiry_logits = self.expiry_head(cls_output)
        security_logits = self.security_head(cls_output)

        card_number_logits = card_number_logits.view(-1, 16, 10)
        expiry_logits = expiry_logits.view(-1, 4, 10)
        security_logits = security_logits.view(-1, 7, 10)

        return card_number_logits, expiry_logits, security_logits

    def predict(self, image):
        self.eval()
        with torch.no_grad():
            card_logits, expiry_logits, security_logits = self.forward(image)
            
            # Compute softmax probabilities
            card_probs = F.softmax(card_logits, dim=-1)
            expiry_probs = F.softmax(expiry_logits, dim=-1)
            security_probs = F.softmax(security_logits, dim=-1)

            # Get predictions
            card_pred = torch.argmax(card_probs, dim=-1).cpu().numpy()[0]
            expiry_pred = torch.argmax(expiry_probs, dim=-1).cpu().numpy()[0]
            security_pred = torch.argmax(security_probs, dim=-1).cpu().numpy()[0]

            # Compute confidence scores (average probability of predicted classes)
            card_confidence = card_probs[0, range(len(card_pred)), card_pred].mean().item()
            expiry_confidence = expiry_probs[0, range(len(expiry_pred)), expiry_pred].mean().item()
            security_confidence = security_probs[0, range(len(security_pred)), security_pred].mean().item()

            card_number = ''.join(map(str, card_pred))
            card_number = ' '.join([card_number[i:i+4] for i in range(0, 16, 4)])
            expiry = ''.join(map(str, expiry_pred))
            expiry = f"{expiry[:2]}/{expiry[2:4]}"
            security = ''.join(map(str, security_pred)).ljust(7, '0')[:7]

            return {
                "predictions": {
                    "card_number": card_number,
                    "expiry_date": expiry,
                    "security_number": security
                },
                "confidence_scores": {
                    "card_number": card_confidence,
                    "expiry_date": expiry_confidence,
                    "security_number": security_confidence
                }
            }

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))