import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess a credit card image for OCR, retaining color.
    Args:
        image_path (str): Path to the input image.
    Returns:
        np.ndarray: Preprocessed color image.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding for contour detection only
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1)

    # Find contours to crop the card
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")

    # Get the largest contour (assumed to be the card)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Validate aspect ratio (credit card is typically ~3.37:2.12)
    aspect_ratio = w / h
    if not (1.2 < aspect_ratio < 2.0):
        raise ValueError("Detected contour does not match credit card aspect ratio")

    # Add buffer to cropping
    x, y = max(0, x-20), max(0, y-20)
    w, h = min(img.shape[1]-x, w+40), min(img.shape[0]-y, h+40)
    
    # Crop the original color image
    cropped = img[y:y+h, x:x+w]

    # Enhance contrast on the color image using CLAHE on each channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced

def save_image(image, output_path):
    """
    Save the processed image.
    Args:
        image (np.ndarray): Image to save.
        output_path (str): Path to save the image.
    """
    cv2.imwrite(output_path, image)