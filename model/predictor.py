import cv2
from PIL import Image
from inference import load_trained_model, predict, SimArgs
from pathlib import Path


CHECKPOINT_PATH = Path("model", "logs", "surveillance", "checkpoint.pth")
DEVICE = "cuda:0"
MODEL = load_trained_model(str(CHECKPOINT_PATH), DEVICE, SimArgs())


def predict_without_inference_time(image):
    """
    Given an OpenCV image, return the predictions in dictionary form.
    """
    # Retrieve model predictions.
    height, width, _ = image.shape
    pil_image = _cv2_to_pil(image)
    preds, _ = predict(MODEL, DEVICE, pil_image, width, height)
    return preds


def predict_with_inference_time(image):
    """
    Given an OpenCV image, return the predictions in dictionary form.
    """
    # Retrieve model predictions.
    height, width, _ = image.shape
    pil_image = _cv2_to_pil(image)
    preds, inference_time = predict(MODEL, DEVICE, pil_image, width, height)
    return preds, inference_time


def _cv2_to_pil(image):
    """
    Convert an OpenCV image to a PIL image.

    Parameter:
    image : OpenCV image, in BGR format.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
