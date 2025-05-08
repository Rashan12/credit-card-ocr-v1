from google.cloud import vision
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Verify the credentials file is set
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not credentials_path or not os.path.exists(credentials_path):
    raise FileNotFoundError(f"Google Cloud credentials file not found at: {credentials_path}")

def extract_card_details(image_path, is_front=True):
    """
    Extract credit card details using Google Vision API.
    Args:
        image_path (str): Path to the preprocessed image.
        is_front (bool): Flag to indicate if the image is the front (True) or back (False).
    Returns:
        dict: Extracted card details (number, expiry, cvv).
    """
    try:
        client = vision.ImageAnnotatorClient()
    except Exception as e:
        return {"error": f"Failed to initialize Vision API client: {str(e)}"}

    if not os.path.exists(image_path):
        return {"error": f"Image file not found at: {image_path}"}

    try:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
    except Exception as e:
        return {"error": f"Failed to read image file: {str(e)}"}

    image = vision.Image(content=content)
    try:
        response = client.text_detection(image=image)
    except Exception as e:
        return {"error": f"Vision API request failed: {str(e)}"}

    if response.error.message:
        return {"error": f"Vision API error: {response.error.message}"}

    texts = response.text_annotations
    if not texts:
        return {"error": "No text detected"}

    result = {
        "card_number": "",
        "expiry_date": "",
        "cvv": ""
    }

    full_text = texts[0].description
    lines = full_text.split('\n')

    # Regular expressions for better matching
    card_number_pattern = re.compile(r'\d{4}\s?\d{4}\s?\d{4}\s?\d{4}')
    expiry_pattern = re.compile(r'\d{2}/\d{2,4}\b')  # Match MM/YY or MM/YYYY at word boundary
    cvv_pattern = re.compile(r'^\d{3,4}$')

    # Search full text for expiry date to handle split or noisy lines
    expiry_match = expiry_pattern.search(full_text)
    if expiry_match:
        result["expiry_date"] = expiry_match.group()

    for line in lines:
        line = line.strip()
        # Match card number
        if card_number_pattern.match(line):
            result["card_number"] = line
        # Match CVV only if it's the back image
        elif not is_front and cvv_pattern.match(line):
            result["cvv"] = line

    return result