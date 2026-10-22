# Japanese CRNN Text Recognition & On-Screen Translator

An end-to-end Japanese Scene Text Recognition (OCR) pipeline built with PyTorch and trained on the ETL Character Database. The repository includes binary dataset extraction tools, synthetic dataset generators, CRNN training with CTC loss, and a real-time On-Screen Translator desktop application.

---

## Features

* Custom CRNN Architecture: Combines deep CNN feature extraction, 1D convolution dimensionality reduction, Bidirectional LSTMs, and Multihead Self-Attention.
* ETL Dataset Extractor: Binary unpacker for ETL character database files (ETL1–ETL9) with dynamic UTF-8 mapping and image scaling.
* Line Image Generators: Automated scripts to generate synthetic horizontal and vertical character line samples for training.
* On-Screen Translator Desktop App: Modern Tkinter GUI with non-blocking async OCR screen region capture, EasyOCR bounding box detection, and live multi-engine translation (translators).

---

## Model Architecture

The model uses a modified Convolutional Recurrent Neural Network (CRNN) designed for sequence-to-sequence character recognition:

1. Feature Extractor: Deep CNN stack (Conv2d + BatchNorm2d + ReLU + MaxPool2d).
2. Dimensionality Reduction: Conv1d layer projecting features to a hidden dimension of 256.
3. Sequence Modeling: Two Bidirectional LSTM layers with multi-head self-attention (MultiheadAttention).
4. Decoder: Linear projection with Connectionist Temporal Classification (CTC) loss decoding.

---

## Project Structure

japanese-crnn-text-recognition/
├── data/                         # Generated vocab and training samples
├── samples/                      # Inference sample image files
├── tools/                        # Extraction utilities and submodules
│   └── etlcdb-image-extractor/   # ETL binary dataset extractor and charset maps
├── .gitignore                    # Git ignore file
├── design.py                     # On-Screen Translator desktop GUI application
├── functions.py                  # Helper functions and CTC output decoder
├── generate_horizontal.py        # Synthetic horizontal line image generator
├── generate_vertical.py          # Synthetic vertical line image generator
├── infer.py                      # Command-line OCR inference script
├── main.py                       # Pipeline execution entry point
├── model.py                      # CRNN PyTorch network architecture definition
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── train.py                      # CRNN training script with CTC loss and validation
└── train_log.txt                 # Training progress log
---

## Getting Started

### Prerequisites

* Python 3.10+
* CUDA 12.4 compatible GPU (recommended for real-time inference and fast training)

### Installation

1. Clone the repository:
git clone [https://github.com/Xreactives/japanese-crnn-text-recognition.git](https://github.com/Xreactives/japanese-crnn-text-recognition.git)
cd japanese-crnn-text-recognition
2. Create and activate a virtual environment:
python -m venv .venv
# On Windows PowerShell:


..venv\Scripts\Activate.ps1
# On Linux/macOS:


source .venv/bin/activate
3. Install dependencies:
pip install -r requirements.txt

---

## Usage

### 1. Extract ETL Dataset

Extract raw ETL binary data into structured PNG image files:
python tools/etlcdb-image-extractor/etl_extractor.py --data_sets 2

### 2. Generate Training Datasets

Generate horizontal or vertical text line samples along with vocabulary mappings (vocab.pkl):
python generate_horizontal.py
python generate_vertical.py

### 3. Train Model

Train the CRNN model on the generated dataset:
python train.py
*Trained checkpoints will be saved to checkpoints/best_crnn.pth.*

### 4. Run Single Image Inference

Predict text from a single line image:
python infer.py --image "samples/test_line.png"

### 5. Launch On-Screen Translator App

Run the desktop capture GUI application:
python design.py

---

## On-Screen Translator Features

* Capture Window: Draggable translucent capture area with real-time text binarization controls (White/Dark text toggle).
* Multi-Engine Translation: Support for Google Translate, Bing, Yandex, DeepL, and Baidu via translators.
* Query & Result Windows: Separate floating sub-windows for input and translated text monitoring.

---

## License

This project is licensed under the MIT License.