import io
import os
import torch
from google.cloud import vision
from google.cloud.vision_v1 import ImageAnnotatorClient
from google.api_core.exceptions import GoogleAPIError
from transformers import ViTModel
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import numpy as np
from vit_model.model.vit_model import CreditCardViT
import base64
from dotenv import load_dotenv
import re
import time
from metrics_db import MetricsDB
from concurrent.futures import TimeoutError
from google.api_core import retry

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not google_credentials or not os.path.exists(google_credentials):
    raise ValueError(f"Invalid GOOGLE_APPLICATION_CREDENTIALS: {google_credentials} not found or inaccessible")
try:
    client = ImageAnnotatorClient.from_service_account_file(google_credentials)
    print("Google Cloud Vision API client initialized successfully at", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
except Exception as e:
    raise ValueError(f"Failed to initialize Vision API client: {str(e)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = CreditCardViT()
pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
vit_model.load_state_dict(pretrained_model.state_dict(), strict=False)
print("Loaded pre-trained weights from google/vit-base-patch16-224")

vit_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api", "vit_model_weights.pth")
if os.path.exists(vit_model_path):
    vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
    print(f"Loaded fine-tuned weights from {vit_model_path}")
else:
    print("No fine-tuned weights found. Using Hugging Face weights.")
vit_model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((96, 96)),  # Reduced from (128, 128) to speed up preprocessing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@retry.Retry(predicate=retry.if_exception_type(GoogleAPIError), deadline=1.5)  # Reduced timeout to 1.5s
def extract_text_from_image(image_content):
    try:
        image = vision.Image(content=image_content)
        response = client.text_detection(image=image)
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        return response.text_annotations[0].description if response.text_annotations else ""
    except GoogleAPIError as e:
        raise Exception(f"Google API error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error in Vision API: {str(e)}")

def process_with_vit(image_tensor):
    with torch.no_grad():
        result = vit_model.predict(image_tensor)
        return result

def live_extract_card_details(image_content, is_front=True):
    side = 'front' if is_front else 'back'
    start_time = time.time()
    try:
        preprocess_start = time.time()
        image = Image.open(io.BytesIO(image_content)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Preprocessing for live_extract ({side}) took {(time.time() - preprocess_start):.2f} seconds")

        vision_start = time.time()
        try:
            text = extract_text_from_image(image_content)
            print(f"Vision API for live_extract ({side}) took {(time.time() - vision_start):.2f} seconds")
            if text:
                lines = text.split('\n')
                card_number = next((l for l in lines if re.match(r'\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}', l)), "") if is_front else ""
                expiry_date = next((l for l in lines if re.match(r'\b(0[1-9]|1[0-2])[/-]([0-9]{2}|[0-9]{4})\b', l)), "") if is_front else ""
                security_number = next((l for l in lines if len(l) in [3, 7] and l.isdigit()), "") if not is_front else ""

                if expiry_date:
                    expiry_date = re.sub(r'[^0-9]', '/', expiry_date)
                    expiry_parts = expiry_date.split('/')
                    if len(expiry_parts) == 2:
                        month = expiry_parts[0][:2]
                        year = expiry_parts[1][-2:] if len(expiry_parts[1]) > 2 else expiry_parts[1]
                        expiry_date = f"{month}/{year}"

                if is_front:
                    if card_number and expiry_date:
                        instruction = "Card detected. Hold steady for a clear scan."
                    elif card_number:
                        instruction = "Card number detected. Adjust position to capture expiry date."
                    elif expiry_date:
                        instruction = "Expiry date detected. Adjust position to capture card number."
                    else:
                        instruction = "No card details detected. Move the card closer and ensure good lighting."
                else:
                    if security_number:
                        instruction = "Security number detected. Hold steady for a clear scan."
                    else:
                        instruction = "No security number detected. Adjust position to capture the CVV."

                ocr_result = {
                    "card_number": card_number,
                    "expiry_date": expiry_date,
                    "security_number": security_number,
                }
            else:
                instruction = "No text detected. Position the card in the frame with good lighting."
                ocr_result = {}
        except Exception as e:
            print(f"Vision API failed in live_extract ({side}): {str(e)}. Falling back to ViT.")
            vit_start = time.time()
            vit_result = process_with_vit(image_tensor)
            print(f"ViT processing for live_extract ({side}) took {(time.time() - vit_start):.2f} seconds")
            card_number = vit_result["predictions"]["card_number"] if is_front else ""
            expiry_date = vit_result["predictions"]["expiry_date"] if is_front else ""
            security_number = vit_result["predictions"]["security_number"] if not is_front else ""

            if is_front:
                if card_number and expiry_date:
                    instruction = "Card detected (ViT). Hold steady for a clear scan."
                elif card_number:
                    instruction = "Card number detected (ViT). Adjust position to capture expiry date."
                elif expiry_date:
                    instruction = "Expiry date detected (ViT). Adjust position to capture card number."
                else:
                    instruction = "No card details detected (ViT). Move the card closer and ensure good lighting."
            else:
                if security_number:
                    instruction = "Security number detected (ViT). Hold steady for a clear scan."
                else:
                    instruction = "No security number detected (ViT). Adjust position to capture the CVV."

            ocr_result = {
                "card_number": card_number,
                "expiry_date": expiry_date,
                "security_number": security_number,
            }

        return {
            "ocr_result": ocr_result,
            "instruction": instruction,
            "processing_time": (time.time() - start_time)
        }
    except Exception as e:
        return {
            "error": str(e),
            "instruction": "Error processing frame. Ensure the card is in frame and well-lit.",
            "processing_time": (time.time() - start_time)
        }

def calculate_accuracy(prediction, ground_truth):
    if not ground_truth or not prediction:
        return 0.0
    total = len(ground_truth)
    correct = sum(1 for p, t in zip(prediction, ground_truth) if p == t)
    return correct / total if total > 0 else 0.0

def extract_card_details(image_content, is_front=True, use_vision_api=True):
    side = 'front' if is_front else 'back'
    start_time = time.time()
    vision_result = None
    vit_result = None
    used_vit = {"card_number": False, "expiry_date": False, "security_number": False}
    confidence_scores = {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0}
    errors = []

    try:
        preprocess_start = time.time()
        image = Image.open(io.BytesIO(image_content)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        snapshot = base64.b64encode(image_content).decode('utf-8')
        print(f"Preprocessing for extract_card_details ({side}) took {(time.time() - preprocess_start):.2f} seconds")

        if use_vision_api:
            vision_start = time.time()
            try:
                text = extract_text_from_image(image_content)
                print(f"Vision API for extract_card_details ({side}) took {(time.time() - vision_start):.2f} seconds")
                if text:
                    lines = text.split('\n')
                    card_number = next((l for l in lines if re.match(r'\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}', l)), "") if is_front else ""
                    expiry_date = next((l for l in lines if re.match(r'\b(0[1-9]|1[0-2])[/-]([0-9]{2}|[0-9]{4})\b', l)), "") if is_front else ""
                    security_number = next((l for l in lines if len(l) in [3, 7] and l.isdigit()), "") if not is_front else ""

                    if expiry_date:
                        expiry_date = re.sub(r'[^0-9]', '/', expiry_date)
                        expiry_parts = expiry_date.split('/')
                        if len(expiry_parts) == 2:
                            month = expiry_parts[0][:2]
                            year = expiry_parts[1][-2:] if len(expiry_parts[1]) > 2 else expiry_parts[1]
                            expiry_date = f"{month}/{year}"
                        else:
                            expiry_date = ""

                    if not card_number and is_front:
                        errors.append("Card number not detected")
                    if not expiry_date and is_front:
                        errors.append("Expiry date not detected")
                    if not security_number and not is_front:
                        errors.append("Security number not detected")

                    vision_result = {
                        "card_number": card_number,
                        "expiry_date": expiry_date,
                        "security_number": security_number,
                        "confidence": 0.9 if not errors else 0.0
                    }
                    confidence_scores.update({
                        "card_number": 0.9 if card_number and is_front else 0.0,
                        "expiry_date": 0.9 if expiry_date and is_front else 0.0,
                        "security_number": 0.9 if security_number and not is_front else 0.0
                    })
                else:
                    errors.append("No text detected by Vision API")
                    vision_result = {"card_number": "", "expiry_date": "", "security_number": ""}
            except TimeoutError:
                print(f"Vision API timed out for {side}. Falling back to ViT.")
                vit_start = time.time()
                vit_result = process_with_vit(image_tensor)
                print(f"ViT processing after timeout ({side}) took {(time.time() - vit_start):.2f} seconds")
                used_vit = {"card_number": is_front, "expiry_date": is_front, "security_number": not is_front}
                confidence_scores = vit_result["confidence_scores"]
                vision_result = {
                    "card_number": vit_result["predictions"]["card_number"] if is_front else "",
                    "expiry_date": vit_result["predictions"]["expiry_date"] if is_front else "",
                    "security_number": vit_result["predictions"]["security_number"] if not is_front else ""
                }
            except Exception as e:
                errors.append(f"Vision API failed: {str(e)}")
                vit_start = time.time()
                vit_result = process_with_vit(image_tensor)
                print(f"ViT processing after error ({side}) took {(time.time() - vit_start):.2f} seconds")
                used_vit = {"card_number": is_front, "expiry_date": is_front, "security_number": not is_front}
                confidence_scores = vit_result["confidence_scores"]
                vision_result = {
                    "card_number": vit_result["predictions"]["card_number"] if is_front else "",
                    "expiry_date": vit_result["predictions"]["expiry_date"] if is_front else "",
                    "security_number": vit_result["predictions"]["security_number"] if not is_front else ""
                }
        else:
            vit_start = time.time()
            vit_result = process_with_vit(image_tensor)
            print(f"ViT processing (no Vision API, {side}) took {(time.time() - vit_start):.2f} seconds")
            used_vit = {"card_number": is_front, "expiry_date": is_front, "security_number": not is_front}
            confidence_scores = vit_result["confidence_scores"]
            vision_result = {
                "card_number": vit_result["predictions"]["card_number"] if is_front else "",
                "expiry_date": vit_result["predictions"]["expiry_date"] if is_front else "",
                "security_number": vit_result["predictions"]["security_number"] if not is_front else ""
            }

        result = {
            "card_number": vision_result["card_number"],
            "expiry_date": vision_result["expiry_date"],
            "security_number": vision_result["security_number"],
            "confidence_scores": confidence_scores,
            "used_vit": used_vit,
            "errors": errors,
            "processing_time": (time.time() - start_time)
        }

        metrics_start = time.time()
        vision_success = 0 if errors else 1
        vit_accuracy_card = vit_accuracy_expiry = vit_accuracy_security = 0.0
        if vision_success and is_front and not use_vision_api:
            if vision_result["card_number"]:
                gt_card = vision_result["card_number"].replace(" ", "")
                pred_card = vit_result["predictions"]["card_number"].replace(" ", "")
                vit_accuracy_card = calculate_accuracy(pred_card, gt_card) if used_vit["card_number"] else 0.0
            if vision_result["expiry_date"]:
                gt_expiry = vision_result["expiry_date"].replace("/", "")
                pred_expiry = vit_result["predictions"]["expiry_date"].replace("/", "")
                vit_accuracy_expiry = calculate_accuracy(pred_expiry, gt_expiry) if used_vit["expiry_date"] else 0.0
        elif vision_success and not is_front and not use_vision_api:
            if vision_result["security_number"]:
                gt_security = vision_result["security_number"]
                pred_security = vit_result["predictions"]["security_number"]
                vit_accuracy_security = calculate_accuracy(pred_security, gt_security) if used_vit["security_number"] else 0.0

        vision_confidence = {
            "card_number": confidence_scores["card_number"] if vision_success and is_front else 0.0,
            "expiry_date": confidence_scores["expiry_date"] if vision_success and is_front else 0.0,
            "security_number": confidence_scores["security_number"] if vision_success and not is_front else 0.0
        }
        vit_confidence = confidence_scores
        metrics_db = MetricsDB()
        metrics_db.log_metrics(
            is_front, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
            vision_confidence["card_number"], vision_confidence["expiry_date"], vision_confidence["security_number"],
            vit_confidence["card_number"], vit_confidence["expiry_date"], vit_confidence["security_number"],
            0, 1 if used_vit["card_number"] else 0, 1 if used_vit["expiry_date"] else 0, 1 if used_vit["security_number"] else 0
        )
        print(f"Metrics logging for extract_card_details ({side}) took {(time.time() - metrics_start):.2f} seconds")

        print(f"Total processing time for extract_card_details ({side}): {(time.time() - start_time):.2f} seconds")
        return result
    except Exception as e:
        errors.append(str(e))
        return {
            "error": str(e),
            "confidence_scores": {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0},
            "used_vit": used_vit,
            "errors": errors,
            "processing_time": (time.time() - start_time)
        }