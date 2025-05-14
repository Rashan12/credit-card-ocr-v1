import io
import os
import torch
from google.cloud import vision
from transformers import ViTModel
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import sqlite3
from datetime import datetime
import numpy as np
from vit_model.model.vit_model import CreditCardViT
from threading import Lock
import base64

# Initialize Vision API client
client = vision.ImageAnnotatorClient()

# Load ViT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = CreditCardViT()
# Load pre-trained weights from Hugging Face
pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
# Assuming CreditCardViT can load ViTModel weights (may need adaptation depending on implementation)
vit_model.load_state_dict(pretrained_model.state_dict(), strict=False)
print("Loaded pre-trained weights from google/vit-base-patch16-224")

vit_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api", "vit_model_weights.pth")
if os.path.exists(vit_model_path):
    vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
    print(f"Loaded fine-tuned weights from {vit_model_path}")
else:
    print("No fine-tuned weights found at specified path. Using Hugging Face weights.")
vit_model.to(device).eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def extract_text_from_image(image_content):
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

def process_with_vit(image):
    with torch.no_grad():
        image_tensor = transform(image).unsqueeze(0).to(device)
        result = vit_model.predict(image)
        return result

class MetricsDB:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsDB, cls).__new__(cls)
                    cls._instance.conn = sqlite3.connect(r"C:\Users\PM_User\Documents\credit-card-ocr\api\metrics.db", check_same_thread=False)
                    cls._instance.initialize()
        return cls._instance

    def initialize(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                side TEXT,
                vision_success INTEGER,
                vit_accuracy_card REAL,
                vit_accuracy_expiry REAL,
                vit_accuracy_security REAL,
                vision_confidence_card REAL,
                vision_confidence_expiry REAL,
                vision_confidence_security REAL,
                vit_confidence_card REAL,
                vit_confidence_expiry REAL,
                vit_confidence_security REAL,
                user_correction INTEGER,
                used_vit_card INTEGER,
                used_vit_expiry INTEGER,
                used_vit_security INTEGER
            )
        ''')
        self.conn.commit()

    def log_metrics(self, is_front, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security, 
                    vision_confidence, vit_confidence, used_vit):
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO metrics (timestamp, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
                                vision_confidence_card, vision_confidence_expiry, vision_confidence_security,
                                vit_confidence_card, vit_confidence_expiry, vit_confidence_security,
                                user_correction, used_vit_card, used_vit_expiry, used_vit_security)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (current_timestamp, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
              vision_confidence["card_number"], vision_confidence["expiry_date"], vision_confidence["security_number"],
              vit_confidence["card_number"], vit_confidence["expiry_date"], vit_confidence["security_number"],
              0, 1 if used_vit["card_number"] else 0, 1 if used_vit["expiry_date"] else 0, 1 if used_vit["security_number"] else 0))
        self.conn.commit()

def calculate_accuracy(prediction, ground_truth):
    if not ground_truth or not prediction:
        return 0.0
    total = len(ground_truth)
    correct = sum(1 for p, t in zip(prediction, ground_truth) if p == t)
    return correct / total if total > 0 else 0.0

def extract_card_details(image_content, is_front=True, use_vision_api=True):
    # Define 'side' based on 'is_front' to fix "name 'side' is not defined" error
    side = 'front' if is_front else 'back'
    
    start_time = datetime.now()
    db_path = r"C:\Users\PM_User\Documents\credit-card-ocr\api\metrics.db"
    vision_result = None
    vit_result = None
    used_vit = {"card_number": False, "expiry_date": False, "security_number": False}
    confidence_scores = {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0}
    errors = []

    try:
        # Convert image content to PIL Image for processing and snapshot
        image = Image.open(io.BytesIO(image_content)).convert('RGB')
        snapshot = base64.b64encode(io.BytesIO(image_content).getvalue()).decode('utf-8')

        # Use Vision API only while use_vision_api is True
        try:
            text = extract_text_from_image(image_content)
            if text:
                lines = text.split('\n')
                card_number = next((l for l in lines if len(l.replace(" ", "")) == 16 and l.replace(" ", "").isdigit()), "")
                expiry_date = next((l for l in lines if "/" in l and len(l.replace("/", "").replace(" ", "")) == 4 and l.replace("/", "").isdigit()), "")
                security_number = next((l for l in lines if len(l) in [3, 4] and l.isdigit()), "") if not is_front else ""

                vision_result = {
                    "card_number": card_number if is_front else "",
                    "expiry_date": expiry_date if is_front else "",
                    "security_number": security_number if not is_front else "",
                    "confidence": 0.9  # Placeholder confidence (adjust based on actual Vision API output)
                }
                confidence_scores = {
                    "card_number": 0.9 if is_front and card_number else 0.0,
                    "expiry_date": 0.9 if is_front and expiry_date else 0.0,
                    "security_number": 0.9 if not is_front and security_number else 0.0
                }
            else:
                raise Exception("No text detected by Vision API")
        except Exception as e:
            print(f"Vision API error: {str(e)}")
            vision_result = {"error": str(e)}

        # Only proceed with ViT if use_vision_api is False (i.e., ViT has reached 100% accuracy)
        if not use_vision_api:
            vit_result = process_with_vit(image)
            used_vit = {
                "card_number": is_front,
                "expiry_date": is_front,
                "security_number": not is_front
            }
            confidence_scores = vit_result["confidence_scores"]
            if is_front:
                vision_result = vision_result or {"card_number": "", "expiry_date": "", "security_number": ""}
                vision_result["card_number"] = vit_result["predictions"]["card_number"]
                vision_result["expiry_date"] = vit_result["predictions"]["expiry_date"]
            else:
                vision_result = vision_result or {"card_number": "", "expiry_date": "", "security_number": ""}
                # Adjust security number length based on card type (placeholder logic)
                security_pred = vit_result["predictions"]["security_number"]
                card_type = "credit" if is_front and vision_result["card_number"] and len(vision_result["card_number"].replace(" ", "")) == 16 else "debit"
                security_number = security_pred[:7] if card_type == "credit" else security_pred[:3]
                vision_result["security_number"] = security_number

        # Prepare final result
        result = {
            "card_number": vision_result["card_number"],
            "expiry_date": vision_result["expiry_date"],
            "security_number": vision_result["security_number"],
            "confidence_scores": confidence_scores,
            "used_vit": used_vit,
            "errors": errors,
            "processing_time": (datetime.now() - start_time).total_seconds()
        }

        # Log metrics (using Vision API result as ground truth if available)
        vision_success = 0 if "error" in vision_result else 1
        vit_accuracy_card = 0.0
        vit_accuracy_expiry = 0.0
        vit_accuracy_security = 0.0
        if vision_success and is_front and not use_vision_api:
            if vision_result["card_number"]:
                gt_card = vision_result["card_number"].replace(" ", "")
                pred_card = vit_result["predictions"]["card_number"].replace(" ", "") if vit_result else ""
                vit_accuracy_card = calculate_accuracy(pred_card, gt_card) if used_vit["card_number"] else 0.0
            if vision_result["expiry_date"]:
                gt_expiry = vision_result["expiry_date"].replace("/", "")
                pred_expiry = vit_result["predictions"]["expiry_date"].replace("/", "") if vit_result else ""
                vit_accuracy_expiry = calculate_accuracy(pred_expiry, gt_expiry) if used_vit["expiry_date"] else 0.0
        elif vision_success and not is_front and not use_vision_api:
            if vision_result["security_number"]:
                gt_security = vision_result["security_number"]
                pred_security = vit_result["predictions"]["security_number"] if vit_result else ""
                vit_accuracy_security = calculate_accuracy(pred_security, gt_security) if used_vit["security_number"] else 0.0

        vision_confidence = {
            "card_number": confidence_scores["card_number"] if vision_success and is_front else 0.0,
            "expiry_date": confidence_scores["expiry_date"] if vision_success and is_front else 0.0,
            "security_number": confidence_scores["security_number"] if vision_success and not is_front else 0.0
        }
        vit_confidence = confidence_scores
        metrics_db = MetricsDB()
        metrics_db.log_metrics(is_front, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security, 
                              vision_confidence, vit_confidence, used_vit)

        return result

    except Exception as e:
        errors.append(str(e))
        return {"error": str(e), "confidence_scores": {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0}, "used_vit": used_vit, "processing_time": (datetime.now() - start_time).total_seconds()}

def calculate_accuracy(prediction, ground_truth):
    if not ground_truth or not prediction:
        return 0.0
    total = len(ground_truth)
    correct = sum(1 for p, t in zip(prediction, ground_truth) if p == t)
    return correct / total if total > 0 else 0.0