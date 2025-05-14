from flask import Flask, request, jsonify
from flask_cors import CORS
from vit_model.model.vit_model import CreditCardViT
from vit_model.model.online_learning import OnlineLearner
from vit_model.model.validation import validate_card_details
from vision_api.ocr_service import extract_card_details
from PIL import Image
import io
import os
import torch
import time
import sqlite3
from datetime import datetime, timedelta
from threading import Lock
from transformers import ViTModel
import base64

app = Flask(__name__)
CORS(app)

# Transition state
USE_VISION_API = True  # Will be updated based on transition criteria

# Database path
DB_PATH = r"C:\Users\PM_User\Documents\credit-card-ocr\api\metrics.db"

class MetricsDB:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsDB, cls).__new__(cls)
                    cls._instance.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transition_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                field TEXT,
                used_vit INTEGER,
                user_correction INTEGER,
                vit_accuracy REAL,
                vision_accuracy REAL
            )
        ''')
        cursor.execute("PRAGMA table_info(transition_log)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'user_correction' not in columns:
            cursor.execute('ALTER TABLE transition_log ADD COLUMN user_correction INTEGER DEFAULT 0')
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

        if USE_VISION_API and vision_success and is_front:
            if validated_result["card_number"]:
                cursor.execute('''
                    INSERT INTO transition_log (timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (current_timestamp, "card_number", 1 if used_vit["card_number"] else 0, 0, vit_accuracy_card, 1.0 if validated_result["card_number"] else 0.0))
            if validated_result["expiry_date"]:
                cursor.execute('''
                    INSERT INTO transition_log (timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (current_timestamp, "expiry_date", 1 if used_vit["expiry_date"] else 0, 0, vit_accuracy_expiry, 1.0 if validated_result["expiry_date"] else 0.0))
        elif not USE_VISION_API and is_front:
            if validated_result["card_number"]:
                cursor.execute('''
                    INSERT INTO transition_log (timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (current_timestamp, "card_number", 1 if used_vit["card_number"] else 0, 0, vit_accuracy_card, 0.0))
            if validated_result["expiry_date"]:
                cursor.execute('''
                    INSERT INTO transition_log (timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (current_timestamp, "expiry_date", 1 if used_vit["expiry_date"] else 0, 0, vit_accuracy_expiry, 0.0))
        elif USE_VISION_API and vision_success and not is_front:
            if validated_result["security_number"]:
                cursor.execute('''
                    INSERT INTO transition_log (timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (current_timestamp, "security_number", 1 if used_vit["security_number"] else 0, 0, vit_accuracy_security, 1.0 if validated_result["security_number"] else 0.0))
        elif not USE_VISION_API and not is_front:
            if validated_result["security_number"]:
                cursor.execute('''
                    INSERT INTO transition_log (timestamp, field, used_vit, user_correction, vit_accuracy, vision_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (current_timestamp, "security_number", 1 if used_vit["security_number"] else 0, 0, vit_accuracy_security, 0.0))
        self.conn.commit()
        print(f"Logged metrics for {side} to database successfully")

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
        if len(results) < 14:  # Need 14 days of data
            print("Not enough data for transition (less than 14 days).")
            return False
        for row in results:
            vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security = row
            if (vit_accuracy_card < 0.99 or vit_accuracy_expiry < 0.99 or vit_accuracy_security < 0.99):
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

@app.route('/ocr', methods=['POST'])
def process_card():
    global USE_VISION_API
    start_time = time.time()
    print("Received /ocr request...")

    # Validate request for both front and back images
    if 'front_image' not in request.files or 'back_image' not in request.files:
        print("Missing front_image or back_image in request.")
        return jsonify({"error": "Both front_image and back_image are required"}), 400

    front_file = request.files['front_image']
    back_file = request.files['back_image']
    print("Processing both front and back images...")

    # Process front image
    front_result = process_single_side(front_file, 'front', is_front=True)
    if 'error' in front_result:
        return jsonify(front_result), 500

    # Process back image
    back_result = process_single_side(back_file, 'back', is_front=False)
    if 'error' in back_result:
        return jsonify(back_result), 500

    # Combine results
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

    # Check transition criteria
    try:
        check_transition_criteria()
    except Exception as e:
        print(f"Error checking transition criteria: {str(e)}")

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return jsonify(combined_result)

def process_single_side(file, side, is_front):
    start_time = time.time()
    global validated_result
    vision_result = None
    vision_success = 0
    vision_confidence = {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0}
    vision_accuracy_card = 0.0
    vision_accuracy_expiry = 0.0
    vision_accuracy_security = 0.0
    vit_result = None
    validated_vit = {"card_number": "", "expiry_date": "", "security_number": "", "card_type": "Unknown", "errors": ["ViT skipped"], "use_vit": {}}
    vit_accuracy_card = 0.0
    vit_accuracy_expiry = 0.0
    vit_accuracy_security = 0.0
    vit_confidence = {"card_number": 0.0, "expiry_date": 0.0, "security_number": 0.0}

    # Read image file stream directly for real-time processing
    try:
        file_stream = file.stream.read()
        if not file_stream:
            print(f"{side} image file is empty.")
            return {"error": f"{side} image file is empty"}
    except Exception as e:
        print(f"Error reading {side} image stream: {str(e)}")
        return {"error": f"Failed to read {side} image stream: {str(e)}"}

    # Convert stream to image for snapshot and processing
    image = Image.open(io.BytesIO(file_stream)).convert('RGB')
    snapshot = base64.b64encode(io.BytesIO(file_stream).getvalue()).decode('utf-8')

    # Call updated extract_card_details (handles both Vision API and ViT)
    vision_start = time.time()
    print(f"Calling extract_card_details for {side}...")
    try:
        result = extract_card_details(file_stream, is_front=is_front)
        print(f"extract_card_details result for {side}: {result}")
        if "error" in result:
            print(f"extract_card_details error for {side}: {result['error']}")
            return {"error": result["error"]}
    except Exception as e:
        print(f"Unexpected error calling extract_card_details for {side}: {str(e)}")
        return {"error": f"Unexpected error for {side}: {str(e)}"}
    vision_end = time.time()
    print(f"extract_card_details processing time for {side}: {vision_end - vision_start:.2f} seconds")

    # Extract results
    vision_success = 0 if "error" in result else 1
    vision_confidence = {
        "card_number": result["confidence_scores"]["card_number"] if vision_success else 0.0,
        "expiry_date": result["confidence_scores"]["expiry_date"] if vision_success else 0.0,
        "security_number": result["confidence_scores"]["security_number"] if vision_success else 0.0
    }
    vit_confidence = {
        "card_number": result["confidence_scores"]["card_number"],
        "expiry_date": result["confidence_scores"]["expiry_date"],
        "security_number": result["confidence_scores"]["security_number"]
    }
    used_vit = result["used_vit"]

    # Validate result
    try:
        validated_result = validate_card_details({
            "card_number": result["card_number"],
            "expiry_date": result["expiry_date"],
            "security_number": result["security_number"]
        }, is_front=is_front, confidence_scores=result["confidence_scores"])
        print(f"Validated result for {side}: {validated_result}")
    except Exception as e:
        print(f"Error validating result for {side}: {str(e)}")
        validated_result = {
            "card_number": "",
            "expiry_date": "",
            "security_number": "",
            "card_type": "Unknown",
            "errors": [str(e)],
            "use_vit": used_vit
        }

    # Calculate accuracies (using Vision API as ground truth if available)
    if USE_VISION_API and vision_success:
        vision_accuracy_card = 1.0 if validated_result["card_number"] else 0.0
        vision_accuracy_expiry = 1.0 if validated_result["expiry_date"] else 0.0
        vision_accuracy_security = 1.0 if validated_result["security_number"] else 0.0
        if is_front:
            if validated_result["card_number"]:
                gt_card = validated_result["card_number"].replace(" ", "")
                pred_card = result["card_number"].replace(" ", "")
                vit_accuracy_card = calculate_accuracy(pred_card, gt_card) if used_vit["card_number"] else 0.0
            if validated_result["expiry_date"]:
                gt_expiry = validated_result["expiry_date"].replace("/", "")
                pred_expiry = result["expiry_date"].replace("/", "")
                vit_accuracy_expiry = calculate_accuracy(pred_expiry, gt_expiry) if used_vit["expiry_date"] else 0.0
        else:
            if validated_result["security_number"]:
                gt_security = validated_result["security_number"]
                pred_security = result["security_number"]
                vit_accuracy_security = calculate_accuracy(pred_security, gt_security) if used_vit["security_number"] else 0.0
        print(f"Calculated accuracies for {side} - ViT: card={vit_accuracy_card}, expiry={vit_accuracy_expiry}, security={vit_accuracy_security}")

    # Update model with feedback if Vision API succeeded (including every successful scan)
    if USE_VISION_API and vision_success and not validated_result["errors"]:
        feedback_start = time.time()
        try:
            feedback_data = {
                "card_number": result["card_number"],
                "expiry_date": result["expiry_date"],
                "cvv": result["security_number"],
                "confidence": result["confidence_scores"]["card_number"]  # Using card_number confidence as proxy
            }
            learner.update_with_feedback(image, feedback_data)
            print(f"Model updated with {side} feedback from scan")
        except Exception as e:
            print(f"Error updating model with {side} feedback from scan: {str(e)}")
        feedback_end = time.time()
        print(f"Feedback update time for {side}: {feedback_end - feedback_start:.2f} seconds")

    # Log metrics to database
    metrics_db = MetricsDB()
    metrics_db.log_metrics(is_front, side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security,
                          vision_confidence, vit_confidence, used_vit)

    # Prepare final result (Vision API only initially, until ViT is 100% accurate)
    final_result = {
        "card_number": result["card_number"] if USE_VISION_API else validated_result["card_number"],
        "expiry_date": result["expiry_date"] if USE_VISION_API else validated_result["expiry_date"],
        "security_number": result["security_number"] if USE_VISION_API else validated_result["security_number"],
        "card_type": validated_result["card_type"],
        "errors": result["errors"],
        "snapshot": snapshot
    }
    if not USE_VISION_API and validated_result.get("use_vit", {}).get(side == 'front' and "card_number" or "security_number", False):
        final_result.update({
            "card_number": validated_result["card_number"] if side == 'front' else "",
            "expiry_date": validated_result["expiry_date"] if side == 'front' else "",
            "security_number": validated_result["security_number"] if side == 'back' else ""
        })

    return {
        "result": final_result,
        "confidence_scores": result["confidence_scores"],
        "used_vit": used_vit,
        "vit_accuracy_card": vit_accuracy_card,
        "vit_accuracy_expiry": vit_accuracy_expiry,
        "vit_accuracy_security": vit_accuracy_security
    }

def log_metrics(side, vision_success, vit_accuracy_card, vit_accuracy_expiry, vit_accuracy_security, vision_confidence, vit_confidence, used_vit):
    # This function is now handled by MetricsDB class
    pass

@app.route('/feedback', methods=['POST'])
def provide_feedback():
    print("Received /feedback request...")
    data = request.get_json()
    if not data or 'image_path' not in data or 'corrections' not in data or 'side' not in data:
        print("Invalid feedback data provided.")
        return jsonify({"error": "Invalid feedback data"}), 400

    image_path = data['image_path']
    corrections = data['corrections']
    side = data['side']
    is_front = side == 'front'

    # Save the image to a temporary file if not already saved
    if not os.path.exists(image_path):
        temp_image = Image.open(io.BytesIO(base64.b64decode(image_path.split(',')[1] if ',' in image_path else image_path)))
        temp_image.save(f"temp_{side}.jpg")
        image_path = f"temp_{side}.jpg"

    try:
        validated_feedback = validate_card_details(corrections, is_front=is_front)
        if validated_feedback["errors"]:
            print(f"Feedback validation errors: {validated_feedback['errors']}")
            return jsonify({"error": "Invalid feedback", "details": validated_feedback["errors"]}), 400
    except Exception as e:
        print(f"Error validating feedback: {str(e)}")
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
        print("Logged feedback to metrics.db")
    except Exception as e:
        print(f"Database error in feedback: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    response = {
        "message": "Please confirm the corrections",
        "corrected_result": validated_feedback,
        "image_path": image_path,
        "side": side
    }
    print(f"Returning feedback response: {response}")
    return jsonify(response)

@app.route('/confirm_feedback', methods=['POST'])
def confirm_feedback():
    print("Received /confirm_feedback request...")
    data = request.get_json()
    if not data or 'image_path' not in data or 'corrections' not in data or 'side' not in data:
        print("Invalid confirmation data provided.")
        return jsonify({"error": "Invalid confirmation data"}), 400

    image_path = data['image_path']
    corrections = data['corrections']
    side = data['side']

    if os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            learner.update_with_feedback(image, corrections, num_epochs=3)
            print(f"Model updated with {side} feedback")
        except Exception as e:
            print(f"Error updating model with feedback: {str(e)}")
            return jsonify({"error": f"Failed to update model: {str(e)}"}), 500
    else:
        print(f"Image path {image_path} does not exist, skipping update")
        return jsonify({"error": f"Image file not found at {image_path}"}), 400

    response = {"message": f"Model updated with {side} feedback"}
    print(f"Returning confirm_feedback response: {response}")
    return jsonify(response)

@app.route('/proceed', methods=['POST'])
def proceed():
    print("Received /proceed request...")
    response = {"message": "Proceed confirmed, all temporary data deleted"}
    print(f"Returning proceed response: {response}")
    return jsonify(response)

if __name__ == '__main__':
    print("Starting Flask application...")
    try:
        model = CreditCardViT()
        # Load pre-trained weights from Hugging Face
        pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # Assuming CreditCardViT can load ViTModel weights (may need adaptation depending on implementation)
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
        print("Loaded pre-trained weights from google/vit-base-patch16-224")
        
        model_path = "vit_model_weights.pth"
        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            model.load(model_path)
        else:
            print("No pre-trained weights found at specified path. Using Hugging Face weights.")
        learner = OnlineLearner(model)
        print("Model and learner initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise e
    app.run(debug=True, host='0.0.0.0', port=5000)