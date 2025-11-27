#!/usr/bin/env python3
"""
Hotdog_NotHotdog.py v1.0.0 - CLIP-based NSFW Detector (2025-11-26)

Scans images and videos for NSFW content using OpenAI CLIP.
Supports rules-based and learned classifier modes.

Requirements: torch, open_clip, pillow, numpy, tqdm, scikit-learn, joblib, opencv-python
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

import open_clip

CONFIG = {
    'nipples': 0.20,
    'genitals': 0.20,
    'anus': 0.20,
    'safe': 0.0
}

PROMPTS = [
    "safe for work image",
    "female nipples",
    "male nipples",
    "penis",
    "vulva",
    "anus",
    "female breast",
    "male chest",
    "bikini",
    "lingerie",
    "cleavage"
]

def load_model():
    """Load the CLIP model, preprocess, and tokenizer."""
    model, preprocess, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    return model, preprocess, tokenizer

def score_image(image_path, model, preprocess, text_features):
    """Score an image against text prompts using CLIP."""
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        similarities = (image_features @ text_features.T).softmax(dim=-1)
    return similarities.squeeze().cpu().numpy()

def score_video_frames(video_path, model, preprocess, text_features, fps=1):
    """Score video frames at specified FPS."""
    cap = cv2.VideoCapture(str(video_path))
    frame_scores = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            scores = score_image_from_pil(pil_image, model, preprocess, text_features)
            frame_scores.append(scores)
        frame_count += 1
    cap.release()
    if frame_scores:
        return np.max(frame_scores, axis=0)
    return np.zeros(len(PROMPTS))

def score_image_from_pil(pil_image, model, preprocess, text_features):
    """Score a PIL image against text prompts."""
    image = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        similarities = (image_features @ text_features.T).softmax(dim=-1)
    return similarities.squeeze().cpu().numpy()

def _policy_decision(scores, config):
    """Decide if content is NSFW based on scores and config."""
    safe_sim = scores[0]
    nipples_sim = max(scores[1], scores[2])
    genitals_sim = max(scores[3], scores[4])
    anus_sim = scores[5]
    explicit_sim = max(nipples_sim, genitals_sim, anus_sim)
    if explicit_sim >= config['nipples'] and explicit_sim > safe_sim:
        return 'nsfw', f"explicit {explicit_sim:.3f} > safe {safe_sim:.3f}"
    return 'sfw', f"safe {safe_sim:.3f} >= explicit {explicit_sim:.3f}"

def build_features_from_scores(scores):
    """Build feature vector from CLIP scores for ML."""
    if len(scores) == 11:
        safe_sim = scores[0]
        nipples_sim = max(scores[1], scores[2])
        genitals_sim = max(scores[3], scores[4])
        anus_sim = scores[5]
        breast_sim = scores[6]
        chest_sim = scores[7]
        clothing_sim = max(scores[8], scores[9], scores[10])
    elif len(scores) == 8:
        safe_sim = scores[0]
        nipples_sim = scores[1]
        penis_sim = scores[2]
        vulva_sim = scores[3]
        anus_sim = scores[4]
        breast_sim = scores[5]
        chest_sim = scores[6]
        clothing_sim = scores[7]
        genitals_sim = max(penis_sim, vulva_sim)
    else:
        raise ValueError("Invalid scores length")
    return [
        safe_sim,
        nipples_sim,
        genitals_sim,
        anus_sim,
        breast_sim - safe_sim,
        chest_sim - safe_sim,
        nipples_sim - clothing_sim,
        genitals_sim - clothing_sim
    ]

def auto_tune(csv_path, model_path, apply=False):
    """Train classifier on CSV data and tune threshold."""
    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = os.path.basename(row['file'])
            if filename.startswith('Faces'):
                label = 0  # SFW
            else:
                label = 1  # NSFW
            scores = [float(row[k]) for k in ['safe_sim', 'nipples_sim', 'penis_sim', 'vulva_sim', 'anus_sim', 'breast_sim', 'chest_sim', 'clothing_sim']]
            features = build_features_from_scores(scores)
            data.append((features, label))
    
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    
    clf = RandomForestClassifier(class_weight={0: 3, 1: 1}, random_state=42)
    clf.fit(X, y)
    
    # Tune threshold
    probs = clf.predict_proba(X)[:, 1]
    best_threshold = 0.5
    best_score = 0
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= threshold).astype(int)
        sfw_correct = ((preds == 0) & (y == 0)).sum()
        nsfw_correct = ((preds == 1) & (y == 1)).sum()
        score = sfw_correct + nsfw_correct * 10  # Penalize SFW false positives
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    if apply:
        persist_new_thresholds({'learned_threshold': best_threshold}, model_path.replace('.pkl', '_thresholds.json'))
        joblib.dump(clf, model_path)
    
    return clf, best_threshold

def persist_new_thresholds(thresholds, path):
    """Save thresholds to JSON file."""
    with open(path, 'w') as f:
        json.dump(thresholds, f)

def iter_images(path):
    """Iterate over image/video files in path."""
    for p in Path(path).rglob('*'):
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov']:
            yield p

def main():
    parser = argparse.ArgumentParser(description="CLIP-based NSFW detector")
    parser.add_argument('input_path', help='Path to images/videos directory')
    parser.add_argument('--out', default='results.csv', help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.20, help='Threshold for rules mode')
    parser.add_argument('--classifier-mode', choices=['rules', 'learned'], default='rules', help='Detection mode')
    parser.add_argument('--classifier-model-path', default='nsfw_classifier.pkl', help='Path to learned model')
    parser.add_argument('--auto-tune-thresholds', action='store_true', help='Auto-tune thresholds')
    parser.add_argument('--apply-auto-tune', action='store_true', help='Apply auto-tuned thresholds')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    if args.auto_tune_thresholds:
        clf, threshold = auto_tune(args.out if Path(args.out).exists() else 'results.csv', args.classifier_model_path, args.apply_auto_tune)
        print(f"Auto-tuned threshold: {threshold}")
        return

    model, preprocess, tokenizer = load_model()
    text_tokens = tokenizer(PROMPTS)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    results = []
    for media_path in tqdm(iter_images(args.input_path)):
        if media_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            scores = score_video_frames(media_path, model, preprocess, text_features)
        else:
            scores = score_image(str(media_path), model, preprocess, text_features)
        
        if args.classifier_mode == 'learned':
            if not Path(args.classifier_model_path).exists():
                print("Learned model not found, falling back to rules")
                decision, reason = _policy_decision(scores, CONFIG)
            else:
                clf = joblib.load(args.classifier_model_path)
                features = build_features_from_scores(scores)
                prob = clf.predict_proba([features])[0][1]
                thresholds_path = args.classifier_model_path.replace('.pkl', '_thresholds.json')
                if Path(thresholds_path).exists():
                    with open(thresholds_path) as f:
                        thresholds = json.load(f)
                    threshold = thresholds.get('learned_threshold', 0.5)
                else:
                    threshold = 0.5
                decision = 'nsfw' if prob >= threshold else 'sfw'
                reason = f"learned prob {prob:.3f}"
        else:
            decision, reason = _policy_decision(scores, {k: args.threshold for k in CONFIG})
        
        result = {
            'file': str(media_path),
            'safe_sim': scores[0],
            'nipples_sim': max(scores[1], scores[2]),
            'penis_sim': scores[3],
            'vulva_sim': scores[4],
            'anus_sim': scores[5],
            'breast_sim': scores[6],
            'chest_sim': scores[7],
            'clothing_sim': max(scores[8], scores[9], scores[10]),
            'nsfw_youtube_guess': decision,
            'reason': reason
        }
        results.append(result)
        if args.verbose or args.debug:
            print(f"{media_path}: {decision} - {reason}")

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {args.out}")

if __name__ == '__main__':
    main()