import os
from google.cloud import vision
from google.cloud.vision_v1 import ImageAnnotatorClient
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv

# Load the .env file from the api directory
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Validate and set GOOGLE_APPLICATION_CREDENTIALS
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not google_credentials:
    print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    exit(1)
if not os.path.exists(google_credentials):
    print(f"Error: GOOGLE_APPLICATION_CREDENTIALS file not found at {google_credentials}")
    exit(1)

# Initialize Vision API client
try:
    print(f"Attempting to initialize Vision API client with credentials: {google_credentials}")
    client = ImageAnnotatorClient.from_service_account_file(google_credentials)
    print("Google Cloud Vision API client initialized successfully!")
except Exception as e:
    print(f"Failed to initialize Vision API client: {str(e)}")
    exit(1)

# Function to test Vision API with a sample image
def test_vision_api(image_path):
    try:
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return

        # Read the image file
        with open(image_path, "rb") as image_file:
            content = image_file.read()

        # Create an image object for Vision API
        image = vision.Image(content=content)

        # Perform text detection
        print("Sending text detection request to Vision API...")
        response = client.text_detection(image=image)

        # Check for errors in the response
        if response.error.message:
            print(f"Vision API error: {response.error.message}")
            return

        # Print the detected text
        if response.text_annotations:
            print("Text detected by Vision API:")
            print(response.text_annotations[0].description)
        else:
            print("No text detected in the image.")
    except GoogleAPIError as e:
        print(f"Google API error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Path to a test image (replace with a valid image path)
    test_image_path = "C:/Users/PM_User/Documents/credit-card-ocr/api/test_image.jpg"  # Update this path
    test_vision_api(test_image_path)