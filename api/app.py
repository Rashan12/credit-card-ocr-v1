from flask import Flask, request, jsonify
from flask_cors import CORS
from vit_model.model.vit_model import CreditCardViT
from vit_model.model.online_learning import OnlineLearner
from vit_model.model.validation import validate_card_details
from vision_api.ocr_service import extract_card_details, live_extract_card_details
from metrics_db import MetricsDB
from PIL import Image
import io
import os
import torch
import time
import base64
import torchvision.transforms as transforms
from dotenv import load_dotenv
import hashlib
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)

USE_VISION_API = True

API_KEYS = {
    "test_key": hashlib.sha256("test_key".encode()).hexdigest()
}

model_trained = False
sample_images = []
sample_labels = [
    {"card_number": "1234567890123456", "expiry_date": "12/25", "cvv": "789"},
]
try:
    model = CreditCardViT()
    pretrained_model = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True)
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    print("Loaded pre-trained weights from google/vit-base-patch16-224")

    model_path = os.path.join(os.path.dirname(__file__), "vit_model_weights.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print(f"Loaded fine-tuned weights from {model_path}")
    metrics_db = MetricsDB()
    learner = OnlineLearner(model, metrics_db)

    if not os.path.exists(model_path) and sample_images:
        print("Performing initial training with sample images...")
        for img_path, label in zip(sample_images, sample_labels):
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                learner.update_with_feedback(img, label, num_epochs=5)
        torch.save(model.state_dict(), model_path)
        model_trained = True
        print("Initial training completed and weights saved.")
    else:
        print("Model and learner initialized successfully.")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    raise

def check_transition_criteria():
    global USE_VISION_API
    try:
        metrics_db = MetricsDB()
        cursor = metrics_db.conn.cursor()
        cursor.execute('''
            SELECT vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security
            FROM metrics
            WHERE timestamp >= datetime('now', '-14 days')
            ORDER BY timestamp
        ''')
        results = cursor.fetchall()
        if len(results) < 14:
            print("Not enough data for transition (less than 14 days).")
            return False
        for row in results:
            vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security = row
            if vit_accuracy_card < 0.99 or vit_accuracy_expiry < 0.99 or vit_accuracy_security < 0.99:
                print("ViT accuracies not high enough for transition.")
                return False
        USE_VISION_API = False
        print("Transition criteria met: Switching to ViT-only mode.")
        return True
    except Exception as e:
        print(f"Error in check_transition_criteria: {str(e)}")
        return False

def calculate_accuracy(prediction, ground_truth):
    if not ground_truth or not prediction:
        return 0.0
    total = len(ground_truth)
    correct = sum(1 for p, t in zip(prediction, ground_truth) if p == t)
    return correct / total if total > 0 else 0.0

@app.route('/docs', methods=['GET'])
def get_docs():
    docs = {
        "endpoints": {
            "/ocr": {
                "method": "POST",
                "description": "Process front and back credit card images for OCR.",
                "headers": {"X-API-Key": "Your API key"},
                "body": {"front_image": "Image file (JPEG/PNG)", "back_image": "Image file (JPEG/PNG)"},
                "response": {
                    "front": {"card_number": "string", "expiry_date": "string", "card_type": "string", "errors": "list", "snapshot": "string"},
                    "back": {"security_number": "string", "errors": "list", "snapshot": "string"},
                    "processing_time": "float"
                }
            },
            "/live_ocr": {
                "method": "POST",
                "description": "Process a single frame for live OCR.",
                "headers": {"X-API-Key": "Your API key"},
                "body": {"image": "Image file (JPEG/PNG)", "is_front": "boolean (true for front, false for back)"},
                "response": {
                    "ocr_result": {"card_number": "string", "expiry_date": "string", "security_number": "string"},
                    "instruction": "string",
                    "processing_time": "float"
                }
            }
        }
    }
    return jsonify(docs)

@app.route('/ocr', methods=['POST'])
def process_card():
    global USE_VISION_API, model_trained
    start_time = time.time()
    print("Received /ocr request...")

    if not model_trained and sample_images:
        print("Performing initial training on first request...")
        for img_path, label in zip(sample_images, sample_labels):
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                learner.update_with_feedback(img, label, num_epochs=5)
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "vit_model_weights.pth"))
        model_trained = True
        print("Initial training completed.")

    if 'front_image' not in request.files or 'back_image' not in request.files:
        return jsonify({"error": "Both front_image and back_image are required"}), 400

    front_file = request.files['front_image']
    back_file = request.files['back_image']
    print("Processing both front and back images...")

    front_result = process_single_side(front_file, 'front', True)
    if 'error' in front_result:
        return jsonify(front_result), 500

    back_result = process_single_side(back_file, 'back', False)
    if 'error' in back_result:
        return jsonify(back_result), 500

    combined_result = {
        "front": front_result["result"],
        "back": back_result["result"],
        "processing_time": time.time() - start_time,
        "confidence_scores": {
            "front": front_result["confidence_scores"],
            "back": back_result["confidence_scores"]
        },
        "used_vit": {
            "front": front_result["used_vit"],
            "back": back_result["used_vit"]
        },
        "metrics": {
            "front": {
                "vision_success": 1 if "error" not in front_result else 0,
                "vit_accuracy_card": front_result.get("vit_accuracy_card", 0.0),
                "vit_accuracy_expiry": front_result.get("vit_accuracy_expiry", 0.0),
                "vit_accuracy_security": front_result.get("vit_accuracy_security", 0.0)
            },
            "back": {
                "vision_success": 1 if "error" not in back_result else 0,
                "vit_accuracy_card": back_result.get("vit_accuracy_card", 0.0),
                "vit_accuracy_expiry": back_result.get("vit_accuracy_expiry", 0.0),
                "vit_accuracy_security": back_result.get("vit_accuracy_security", 0.0)
            }
        }
    }
    print(f"Combined result: {combined_result}")

    try:
        check_transition_criteria()
    except Exception as e:
        print(f"Error checking transition criteria: {str(e)}")

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return jsonify(combined_result)

def process_single_side(file, side, is_front):
    start_time = time.time()
    print(f"Starting processing for {side} side at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    read_start = time.time()
    try:
        file_stream = file.stream.read()
        if not file_stream:
            return {
                "error": f"{side} image file is empty",
                "confidence_scores": {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0},
                "used_vit": {"card_number": False, "expiry_date": False, "security_number": False}
            }
    except Exception as e:
        return {
            "error": f"Failed to read {side} image stream: {str(e)}",
            "confidence_scores": {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0},
            "used_vit": {"card_number": False, "expiry_date": False, "security_number": False}
        }
    print(f"File read for {side} took {(time.time() - read_start):.2f} seconds")

    image_open_start = time.time()
    try:
        image = Image.open(io.BytesIO(file_stream)).convert('RGB')
        snapshot = base64.b64encode(file_stream).decode('utf-8')
        cropped_snapshot = snapshot
    except Exception as e:
        return {
            "error": f"Failed to open {side} image: {str(e)}",
            "confidence_scores": {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0},
            "used_vit": {"card_number": False, "expiry_date": False, "security_number": False}
        }
    print(f"Image open and encode for {side} took {(time.time() - image_open_start):.2f} seconds")

    extract_start = time.time()
    try:
        result = extract_card_details(file_stream, is_front, USE_VISION_API)
        print(f"Result from extract_card_details for {side}: {result}")
        if not isinstance(result, dict):
            return {
                "error": f"Invalid result from extract_card_details for {side}: {result}",
                "confidence_scores": {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0},
                "used_vit": {"card_number": False, "expiry_date": False, "security_number": False}
            }
        if "error" in result:
            return {
                "error": result["error"],
                "confidence_scores": result.get("confidence_scores", {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0}),
                "used_vit": result.get("used_vit", {"card_number": False, "expiry_date": False, "security_number": False})
            }
    except Exception as e:
        return {
            "error": f"Processing error for {side} in extract_card_details: {str(e)}",
            "confidence_scores": {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0},
            "used_vit": {"card_number": False, "expiry_date": False, "security_number": False}
        }
    print(f"Extract card details for {side} took {(time.time() - extract_start):.2f} seconds")

    validate_start = time.time()
    vision_success = 0 if "error" in result else 1
    vision_confidence = {
        "card_number": result.get("confidence_scores", {}).get("card_number", 0.0) if vision_success else 0.0,
        "expiry_date": result.get("confidence_scores", {}).get("expiry_date", 0.0) if vision_success else 0.0,
        "security_number": result.get("confidence_scores", {}).get("security_number", 0.0) if vision_success else 0.0
    }
    vit_confidence = result.get("confidence_scores", {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0})
    used_vit = result.get("used_vit", {"card_number": False, "expiry_date": False, "security_number": False})

    try:
        validated_result = validate_card_details({
            "card_number": result.get("card_number", ""),
            "expiry_date": result.get("expiry_date", ""),
            "security_number": result.get("security_number", "")
        }, is_front, result.get("confidence_scores", {}))
    except Exception as e:
        return {
            "error": f"Validation error for {side}: {str(e)}",
            "confidence_scores": {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0},
            "used_vit": {"card_number": False, "expiry_date": False, "security_number": False}
        }
    print(f"Validation for {side} took {(time.time() - validate_start):.2f} seconds")

    # Log ViT predictions for monitoring (even if not used)
    vit_predictions = learner.model.predict(image)
    metrics_db = MetricsDB()
    metrics_db.log_transition(
        "card_number" if is_front else "security_number",
        1 if used_vit["card_number"] or used_vit["security_number"] else 0,
        0,
        vit_confidence["card_number"] if is_front else vit_confidence["security_number"],
        vision_confidence["card_number"] if is_front else vision_confidence["security_number"]
    )

    feedback_start = time.time()
    if USE_VISION_API and vision_success:
        feedback_data = {
            "card_number": validated_result.get("card_number", ""),
            "expiry_date": validated_result.get("expiry_date", ""),
            "cvv": validated_result.get("security_number", ""),
            "confidence": result.get("confidence_scores", {}).get("card_number", 0.0)
        }
        if feedback_data["card_number"] or feedback_data["expiry_date"] or feedback_data["cvv"]:
            try:
                learner.update_with_feedback(image, feedback_data)
            except Exception as e:
                print(f"Error updating learner with feedback for {side}: {str(e)}")
    print(f"Feedback update for {side} took {(time.time() - feedback_start):.2f} seconds")

    metrics_start = time.time()
    vit_accuracy_card = vit_accuracy_expiry = vit_accuracy_security = 0.0
    if USE_VISION_API and vision_success and is_front:
        if validated_result.get("card_number"):
            gt_card = validated_result["card_number"].replace(" ", "")
            pred_card = result.get("card_number", "").replace(" ", "")
            vit_accuracy_card = calculate_accuracy(pred_card, gt_card) if used_vit["card_number"] else 0.0
        if validated_result.get("expiry_date"):
            gt_expiry = validated_result["expiry_date"].replace("/", "")
            pred_expiry = result.get("expiry_date", "").replace("/", "")
            vit_accuracy_expiry = calculate_accuracy(pred_expiry, gt_expiry) if used_vit["expiry_date"] else 0.0
    elif USE_VISION_API and vision_success and not is_front:
        if validated_result.get("security_number"):
            gt_security = validated_result["security_number"]
            pred_security = result.get("security_number", "")
            vit_accuracy_security = calculate_accuracy(pred_security, gt_security) if used_vit["security_number"] else 0.0

    metrics_db.log_metrics(
        is_front, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
        vision_confidence["card_number"], vision_confidence["expiry_date"], vision_confidence["security_number"],
        vit_confidence["card_number"], vit_confidence["expiry_date"], vit_confidence["security_number"],
        0, 1 if used_vit["card_number"] else 0, 1 if used_vit["expiry_date"] else 0, 1 if used_vit["security_number"] else 0
    )
    print(f"Metrics logging for {side} took {(time.time() - metrics_start):.2f} seconds")

    final_result = {
        "card_number": result.get("card_number", ""),
        "expiry_date": result.get("expiry_date", ""),
        "security_number": result.get("security_number", ""),
        "card_type": validated_result.get("card_type", ""),
        "errors": result.get("errors", []),
        "snapshot": cropped_snapshot
    }
    if not USE_VISION_API:
        final_result.update({
            "card_number": validated_result.get("card_number", "") if side == 'front' else "",
            "expiry_date": validated_result.get("expiry_date", "") if side == 'front' else "",
            "security_number": validated_result.get("security_number", "") if side == 'back' else ""
        })

    print(f"Total processing time for {side}: {(time.time() - start_time):.2f} seconds")
    return {
        "result": final_result,
        "confidence_scores": result.get("confidence_scores", {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0}),
        "used_vit": used_vit,
        "vit_accuracy_card": vit_accuracy_card,
        "vit_accuracy_expiry": vit_accuracy_expiry,
        "vit_accuracy_security": vit_accuracy_security
    }

@app.route('/live_ocr', methods=['POST'])
def live_ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']
    image_content = image_file.read()
    is_front = request.form.get('is_front', 'true').lower() == 'true'
    try:
        result = live_extract_card_details(image_content, is_front)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Live OCR error: {str(e)}", "processing_time": 0.0}), 500

@app.route('/feedback', methods=['POST'])
def provide_feedback():
    data = request.get_json()
    if not data or 'image_path' not in data or 'corrections' not in data or 'side' not in data:
        return jsonify({"error": "Invalid feedback data"}), 400

    image_path = data['image_path']
    corrections = data['corrections']
    side = data['side']
    is_front = side == 'front'

    if not os.path.exists(image_path):
        try:
            temp_image = Image.open(io.BytesIO(base64.b64decode(image_path.split(',')[1] if ',' in image_path else image_path)))
            temp_image.save(f"temp_{side}.jpg")
            image_path = f"temp_{side}.jpg"
        except Exception as e:
            return jsonify({"error": f"Failed to save temporary image for feedback: {str(e)}"}), 500

    try:
        validated_feedback = validate_card_details(corrections, is_front)
        if validated_feedback.get("errors"):
            return jsonify({"error": "Invalid feedback", "details": validated_feedback["errors"]}), 400
    except Exception as e:
        return jsonify({"error": f"Feedback validation failed: {str(e)}"}), 500

    try:
        metrics_db = MetricsDB()
        cursor = metrics_db.conn.cursor()
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
        cursor.execute('''
            INSERT INTO metrics (timestamp, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
                                vision_confidence_card, vision_confidence_expiry, vision_confidence_security,
                                vit_confidence_card, vit_confidence_expiry, vit_confidence_security,
                                user_correction, used_vit_card, used_vit_expiry, used_vit_security)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), side, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 0, 0))
        metrics_db.conn.commit()
    except Exception as e:
        return jsonify({"error": f"Database error in feedback: {str(e)}"}), 500

    return jsonify({
        "message": "Please confirm the corrections",
        "corrected_result": validated_feedback,
        "image_path": image_path,
        "side": side
    })

@app.route('/confirm_feedback', methods=['POST'])
def confirm_feedback():
    data = request.get_json()
    if not data or 'image_path' not in data or 'corrections' not in data or 'side' not in data:
        return jsonify({"error": "Invalid confirmation data"}), 400

    image_path = data['image_path']
    corrections = data['corrections']
    side = data['side']

    if os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            learner.update_with_feedback(image, corrections, num_epochs=3)
        except Exception as e:
            return jsonify({"error": f"Failed to update model: {str(e)}"}), 500
    else:
        return jsonify({"error": f"Image file not found at {image_path}"}), 400

    return jsonify({"message": f"Model updated with {side} feedback"})

@app.route('/proceed', methods=['POST'])
def proceed():
    try:
        learner.process_feedback_batch()
        # Delete temporary image files
        for side in ['front', 'back']:
            temp_file = f"temp_{side}.jpg"
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Deleted temporary file: {temp_file}")
    except Exception as e:
        print(f"Error processing feedback buffer: {str(e)}")
        return jsonify({"error": f"Failed to process feedback: {str(e)}"}), 500
    return jsonify({"message": "Proceed confirmed, all temporary data deleted"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)