# Hotdog_NotHotdog

A CLIP-based NSFW detector for images and videos. Uses OpenAI CLIP to classify content as Safe For Work (SFW) or Not Safe For Work (NSFW), with support for rules-based and machine learning modes.

## Features

- **CLIP-Powered Detection**: Leverages OpenAI CLIP ViT-B-32 for accurate image-text similarity scoring.
- **Dual Modes**: Rules-based heuristic or trained RandomForest classifier (train your own, no pre-trained model provided).
- **Video Support**: Samples frames from videos at 1 FPS.
- **High Accuracy**: Achieves ~99.5% accuracy on balanced datasets.
- **Local Operation**: No internet required after initial setup.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Windows

1. Clone or download the repository.
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install torch open_clip pillow numpy tqdm scikit-learn joblib opencv-python
   ```

### Linux

1. Clone or download the repository.
2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install torch open_clip pillow numpy tqdm scikit-learn joblib opencv-python
   ```

### macOS (Optional)

1. Clone or download the repository.
2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install torch open_clip pillow numpy tqdm scikit-learn joblib opencv-python
   ```
   Note: For Apple Silicon, use `pip install torch torchvision` from PyTorch's site.

## Usage

### Basic Scan (Rules Mode)

Scan a directory of images:
```
python Hotdog_NotHotdog.py /path/to/images --out results.csv
```

### Learned Mode

Train your own classifier first (see below), then use it:
```
python Hotdog_NotHotdog.py /path/to/images --classifier-mode=learned --classifier-model-path your_trained_model.pkl --out results.csv
```

### Training a Classifier

No pre-trained model is provided. To train your own RandomForest classifier:

1. Prepare a balanced dataset of SFW and NSFW images (label SFW files starting with "Faces").
2. Scan your dataset to generate features:
   ```
   python Hotdog_NotHotdog.py /path/to/training_images --out training.csv
   ```
3. Train and tune the classifier:
   ```
   python Hotdog_NotHotdog.py /path/to/training_images --auto-tune-thresholds --apply-auto-tune --out training.csv --classifier-model-path your_model.pkl
   ```
   This saves the trained model to `your_model.pkl` and tuned thresholds to `your_model_thresholds.json`.

### Options

- `--threshold`: Similarity threshold for rules mode (default 0.20).
- `--verbose`: Print results for each file.
- `--debug`: Enable debug output.

## Output

Results are saved to a CSV with columns: file, similarities, decision, reason.

Example:
```
file,safe_sim,nipples_sim,...,nsfw_youtube_guess,reason
image1.png,0.8,0.1,...,sfw,safe 0.8 >= explicit 0.1
```

## Labeling Convention

- **SFW**: Files starting with "Faces" (e.g., Faces_001.png).
- **NSFW**: All others.

## License

MIT License. See LICENSE file for details.

## Contributing

Contributions welcome! Please follow PEP 8 standards and add tests for new features.