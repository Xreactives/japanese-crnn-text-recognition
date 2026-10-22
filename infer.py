import argparse
import os
import pickle
import easyocr
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from functions import decode_ctc_output
from model import CRNN


def binarize_image(image: Image.Image, threshold: int = 183, invert: bool = False) -> Image.Image:
    """Converts a PIL image to grayscale and applies thresholding."""
    grayscale = image.convert("L")
    if invert:
        return grayscale.point(lambda x: 255 if x < threshold else 0, "1").convert("L")
    return grayscale.point(lambda x: 0 if x < threshold else 255, "1").convert("L")


def detect_and_crop_lines(image_path: str, reader: easyocr.Reader, transform, threshold: int = 183):
    """Detects text regions via EasyOCR, crops them, and applies transformations."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found at: {image_path}")

    raw_image = Image.open(image_path)
    processed_image = binarize_image(raw_image, threshold=threshold)

    results = reader.readtext(image_path)

    # Fallback to whole image if EasyOCR detects no text bounding boxes
    if not results:
        return [transform(processed_image)]

    tensor_images = []
    for bbox, _, _ in results:
        # Bounding box coordinates: [top-left, top-right, bottom-right, bottom-left]
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # Crop text line region
        cropped = processed_image.crop((*top_left, *bottom_right))
        tensor_images.append(transform(cropped))

    return tensor_images


def predict_batch(model, tensor_images, vocab, device):
    """Runs batch inference on preprocessed line image tensors."""
    model.eval()

    # Determine maximum width in the sequence for batch padding
    max_width = max(t.shape[2] for t in tensor_images)
    padded_tensors = []

    for t in tensor_images:
        pad_width = max_width - t.shape[2]
        # Pad right side of tensor image: (left, right, top, bottom)
        padded = F.pad(t, (0, pad_width, 0, 0), value=0.0)
        padded_tensors.append(padded)

    batch_tensor = torch.stack(padded_tensors).to(device)  # [B, 1, H, W]

    with torch.inference_mode():
        outputs = model(batch_tensor)  # Forward pass -> [B, W, num_classes]
        outputs = outputs.permute(1, 0, 2)  # [seq_len, B, num_classes]
        log_probs = F.log_softmax(outputs, dim=2)

        predictions = decode_ctc_output(log_probs, vocab)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Japanese CRNN Text Recognition Inference Pipeline")
    parser.add_argument("--image", type=str, default="samples/2.jpg", help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_crnn.pth", help="Path to model checkpoint")
    parser.add_argument("--threshold", type=int, default=183, help="Binarization threshold (0-255)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model & Vocabulary from Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        return

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vocab = checkpoint["vocab"]

    model = CRNN(input_height=32, num_classes=len(vocab)).to(device)
    model.load_state_dict(checkpoint["model"])

    # Define standard image transformation pipeline
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 128)),  # Fix height to 32, resize width
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    print("Initializing EasyOCR text detector...")
    reader = easyocr.Reader(["ja"], gpu=torch.cuda.is_available())

    print(f"Processing image: {args.image}")
    tensor_images = detect_and_crop_lines(args.image, reader, transform, threshold=args.threshold)

    print("\n--- OCR Inference Results ---")
    results = predict_batch(model, tensor_images, vocab, device)
    for idx, text in enumerate(results):
        print(f"Line {idx + 1}: {text}")


if __name__ == "__main__":
    main()