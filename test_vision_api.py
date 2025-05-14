from google.cloud import vision
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def test_vision_api(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        print(f"Error: {response.error.message}")
    else:
        print("Success! Detected text:")
        print(response.text_annotations[0].description)

if __name__ == "__main__":
    test_image_path = "test_card.jpg"  
    test_vision_api(test_image_path)