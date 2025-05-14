from vit_model import CreditCardViT
from online_learning import OnlineLearner

# Initialize model and learner
model = CreditCardViT()
learner = OnlineLearner(model)

# Simulate user feedback
user_feedback = {
    "card_number": "4283 9820 0208 7152",
    "expiry_date": "04/26"
}

# Update model with feedback
learner.update_with_feedback('processed_card.jpg', user_feedback)