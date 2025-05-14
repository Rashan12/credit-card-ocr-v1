from vit_model import CreditCardViT
from PIL import Image

# Initialize model
model = CreditCardViT()

# Load preprocessed image
image = Image.open('processed_card.jpg').convert('RGB')

# Predict
result = model.predict(image)
print(result)