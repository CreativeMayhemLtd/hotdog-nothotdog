#!/usr/bin/env python3
"""
<<<<<<< HEAD
Hotdog_NotHotdog.py v0.1.0 - CLIP-based NSFW Detector (2025-11-26)
=======
Hotdog_NotHotdog.py v0.1.1 - CLIP-based NSFW Detector (2025-11-27)
>>>>>>> 9c28884 (chore: update gitignore and legacy Hotdog_NotHotDog monitor)

Scans images and videos for NSFW content using OpenAI CLIP.
Supports rules-based and learned classifier modes.

This version adds optional GPU support, batched scoring, improved
video sampling, and a more robust learned mode while keeping
backwards-compatible CLI flags and CSV output when requested.

Requirements: torch, open_clip, pillow, numpy, tqdm, scikit-learn,
joblib, opencv-python
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
    "nipples": 0.20,
    "genitals": 0.20,
    "anus": 0.20,
    "safe": 0.0,
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
    "cleavage",
]

FEATURE_VERSION = "v2"


def load_model(model_name="ViT-L-14", pretrained="laion2b_s32b_b82k", device=None, legacy_model=False):
    """Load the CLIP model, preprocess, tokenizer, and device.

    legacy_model=True forces the original v0.1.0 backbone.
    """

    if legacy_model:
        model_name = "ViT-B-32"
        pretrained = "laion2b_s34b_b79k"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)

    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device_obj,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer, device_obj


@torch.inference_mode()
def encode_prompts(prompts, model, tokenizer, device):
    """Encode text prompts once and normalize features."""

    tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.inference_mode()
def score_pil_batch(pil_images, model, preprocess, text_features, device, batch_size=32):
    """Score a batch of PIL images against prompts using CLIP.

    Returns a NumPy array of shape (N, len(PROMPTS)).
    """

    if not pil_images:
        return np.zeros((0, len(PROMPTS)), dtype=np.float32)

    scores = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        images = [preprocess(img.convert("RGB")) for img in batch]
        image_tensor = torch.stack(images).to(device)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).softmax(dim=-1)
        scores.extend(similarities.cpu().numpy())
    return np.asarray(scores, dtype=np.float32)


def score_image_path(image_path, model, preprocess, text_features, device, batch_size=32):
    """Score a single image path using batched helper."""

    pil_image = Image.open(image_path).convert("RGB")
    scores = score_pil_batch([pil_image], model, preprocess, text_features, device, batch_size=batch_size)
    return scores[0]


def score_video_frames(video_path, model, preprocess, text_features, device, fps=1.0, batch_size=32, max_frames=200):
    """Score video frames at specified FPS and aggregate by max.

    This samples frames approximately at the requested FPS and uses
    the most explicit frame (per-prompt max) for the video score.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros(len(PROMPTS), dtype=np.float32)

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 1.0
    step = max(1, int(actual_fps / fps))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            if len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not frames:
        return np.zeros(len(PROMPTS), dtype=np.float32)

    frame_scores = score_pil_batch(frames, model, preprocess, text_features, device, batch_size=batch_size)
    return frame_scores.max(axis=0)

def _policy_decision(scores, config):
    """Decide if content is NSFW based on scores and config.

    Policy is unchanged from v0.1.0: content is NSFW if any explicit
    channel (nipples, genitals, anus) is above the threshold and
    greater than the safe score.
    """

    safe_sim = float(scores[0])
    nipples_sim = float(max(scores[1], scores[2]))
    genitals_sim = float(max(scores[3], scores[4]))
    anus_sim = float(scores[5])
    explicit_sim = max(nipples_sim, genitals_sim, anus_sim)
    if explicit_sim >= config["nipples"] and explicit_sim > safe_sim:
        return "nsfw", f"explicit {explicit_sim:.3f} > safe {safe_sim:.3f}"
    return "sfw", f"safe {safe_sim:.3f} >= explicit {explicit_sim:.3f}"

def build_features_from_scores(scores):
    """Build feature vector from CLIP scores for ML.

    FEATURE_VERSION v2 expands slightly on the original engineering
    while remaining backwards compatible with v0.1.0 CSVs.
    """

    if len(scores) == 11:
        safe_sim = float(scores[0])
        nipples_sim = float(max(scores[1], scores[2]))
        genitals_sim = float(max(scores[3], scores[4]))
        anus_sim = float(scores[5])
        breast_sim = float(scores[6])
        chest_sim = float(scores[7])
        clothing_sim = float(max(scores[8], scores[9], scores[10]))
    elif len(scores) == 8:
        safe_sim = float(scores[0])
        nipples_sim = float(scores[1])
        penis_sim = float(scores[2])
        vulva_sim = float(scores[3])
        anus_sim = float(scores[4])
        breast_sim = float(scores[5])
        chest_sim = float(scores[6])
        clothing_sim = float(scores[7])
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
        genitals_sim - clothing_sim,
        anus_sim - clothing_sim,
    ]

def auto_tune(csv_path, model_path, apply=False):
    """Train classifier on CSV data and tune threshold.

    Supports both legacy v0.1.0 CSVs and newer v0.2.0 layouts.
    """

    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = os.path.basename(row["file"])
            label = 0 if filename.startswith("Faces") else 1

            # Legacy schema: single nipples/clothing columns
            if "nipples_sim" in row and "clothing_sim" in row:
                scores = [
                    float(row.get("safe_sim", 0.0)),
                    float(row.get("nipples_sim", 0.0)),
                    float(row.get("penis_sim", 0.0)),
                    float(row.get("vulva_sim", 0.0)),
                    float(row.get("anus_sim", 0.0)),
                    float(row.get("breast_sim", 0.0)),
                    float(row.get("chest_sim", 0.0)),
                    float(row.get("clothing_sim", 0.0)),
                ]
            else:
                # Future schema: explicit per-prompt scores
                scores = [
                    float(row.get("safe_sim", 0.0)),
                    float(row.get("female_nipples_sim", row.get("nipples_sim", 0.0))),
                    float(row.get("male_nipples_sim", 0.0)),
                    float(row.get("penis_sim", 0.0)),
                    float(row.get("vulva_sim", 0.0)),
                    float(row.get("anus_sim", 0.0)),
                    float(row.get("breast_sim", 0.0)),
                    float(row.get("chest_sim", 0.0)),
                    float(row.get("bikini_sim", 0.0)),
                    float(row.get("lingerie_sim", 0.0)),
                    float(row.get("cleavage_sim", 0.0)),
                ]

            features = build_features_from_scores(scores)
            data.append((features, label))

    if not data:
        raise ValueError("No training data found in CSV for auto-tune")

    X = np.array([d[0] for d in data], dtype=np.float32)
    y = np.array([d[1] for d in data], dtype=np.int32)

    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight={0: 2, 1: 1},
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Tune threshold with strong penalty on SFW false positives
    probs = clf.predict_proba(X)[:, 1]
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= threshold).astype(int)
        sfw_correct = ((preds == 0) & (y == 0)).sum()
        nsfw_correct = ((preds == 1) & (y == 1)).sum()
        sfw_incorrect = ((preds == 1) & (y == 0)).sum()
        nsfw_incorrect = ((preds == 0) & (y == 1)).sum()
        # Slightly relax SFW weighting and penalize NSFW misses more,
        # to buy back some recall on explicit content while still
        # strongly discouraging SFW false positives.
        score = sfw_correct * 8 + nsfw_correct * 2 - (sfw_incorrect * 15 + nsfw_incorrect * 8)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    if apply:
        model_path = Path(model_path)
        thresholds_path = model_path.with_name(model_path.stem + "_thresholds.json")
        persist_new_thresholds(
            {"learned_threshold": best_threshold, "feature_version": FEATURE_VERSION},
            thresholds_path,
        )
        joblib.dump(clf, str(model_path))

    return clf, best_threshold

def persist_new_thresholds(thresholds, path):
    """Save thresholds to JSON file."""
    with open(path, 'w') as f:
        json.dump(thresholds, f)

def iter_media_files(path):
    """Iterate over image/video files in path."""

    exts = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".avi", ".mov", ".mkv"}
    for p in Path(path).rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def main():
    parser = argparse.ArgumentParser(description="CLIP-based NSFW detector v0.2.0")
    parser.add_argument("input_path", help="Path to images/videos directory")
    parser.add_argument("--out", default="results.csv", help="Output CSV file")
    parser.add_argument("--threshold", type=float, default=0.20, help="Threshold for rules mode")

    # New primary mode flag, plus legacy alias
    parser.add_argument("--mode", choices=["rules", "learned"], default="rules", help="Detection mode")
    parser.add_argument(
        "--classifier-mode",
        choices=["rules", "learned"],
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--classifier-model-path",
        "--classifier",
        dest="classifier_model_path",
        default="nsfw_classifier.pkl",
        help="Path to learned model",
    )

    parser.add_argument(
        "--training-csv",
        dest="training_csv",
        help=(
            "CSV file to use for --auto-tune. "
            "If omitted, falls back to an existing --out/results.csv."
        ),
    )

    # Auto-tune flags (new + legacy aliases)
    parser.add_argument("--auto-tune", action="store_true", help="Auto-tune thresholds for learned mode")
    parser.add_argument("--apply-tune", action="store_true", help="Persist auto-tuned classifier and thresholds")
    parser.add_argument("--auto-tune-thresholds", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--apply-auto-tune", action="store_true", help=argparse.SUPPRESS)

    # Model / device / batching
    parser.add_argument("--model", default="ViT-L-14", help="CLIP model name")
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k", help="Pretrained weights identifier")
    parser.add_argument("--device", default=None, help="Device to use (cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP inference")
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling FPS")
    parser.add_argument("--legacy-model", action="store_true", help="Use original v0.1.0 CLIP backbone")
    parser.add_argument("--legacy-csv", action="store_true", help="Emit legacy v0.1.0 CSV schema")

    parser.add_argument("--sanity-check", action="store_true", help="Run built-in sanity checks and exit")
    parser.add_argument(
        "--sanity-path",
        default=None,
        help="Optional path for sanity check set (defaults to test_images/ if present)",
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    args = parser.parse_args()

    if args.version:
        print("Hotdog_NotHotdog.py v0.1.1")
        return

    # Legacy flag mapping
    if args.classifier_mode is not None:
        args.mode = args.classifier_mode
    if args.auto_tune_thresholds:
        args.auto_tune = True
    if args.apply_auto_tune:
        args.apply_tune = True

    if args.auto_tune:
        if args.training_csv is not None:
            csv_path = Path(args.training_csv)
        else:
            csv_path = Path(args.out) if Path(args.out).exists() else Path("results.csv")

        clf, threshold = auto_tune(csv_path, Path(args.classifier_model_path), args.apply_tune)
        print(f"Auto-tuned threshold: {threshold:.3f}")
        return

    if args.sanity_check:
        sanity_root = None
        if args.sanity_path is not None:
            sanity_root = Path(args.sanity_path)
        else:
            default_path = Path("test_images")
            if default_path.exists():
                sanity_root = default_path
        if sanity_root is None or not sanity_root.exists():
            raise SystemExit("Sanity check requested but no valid --sanity-path and no test_images/ found")

        # For sanity we require learned mode and a trained classifier
        model_path = Path(args.classifier_model_path)
        if not model_path.exists():
            raise SystemExit("Sanity check requires a trained classifier model; none found at --classifier-model-path")

        try:
            thresholds_path = model_path.with_name(model_path.stem + "_thresholds.json")
            with open(thresholds_path) as f:
                thresholds_data = json.load(f)
            learned_threshold = float(thresholds_data.get("learned_threshold", 0.5))
        except FileNotFoundError:
            learned_threshold = 0.5

        # Load model and prompts once
        model, preprocess, tokenizer, device = load_model(
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
            legacy_model=args.legacy_model,
        )
        text_features = encode_prompts(PROMPTS, model, tokenizer, device)

        clf = joblib.load(str(model_path))

        files = list(iter_media_files(sanity_root))
        if not files:
            raise SystemExit("Sanity check set is empty")

        faces_sfw_rules = 0
        faces_total = 0
        nsfw_hits_learned = 0

        for media_path in files:
            is_video = media_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            if is_video:
                scores = score_video_frames(
                    media_path,
                    model,
                    preprocess,
                    text_features,
                    device,
                    fps=args.fps,
                    batch_size=args.batch_size,
                )
            else:
                scores = score_image_path(
                    media_path,
                    model,
                    preprocess,
                    text_features,
                    device,
                    batch_size=args.batch_size,
                )

            decision_rules, _ = _policy_decision(scores, {**CONFIG, "nipples": args.threshold})

            features = build_features_from_scores(scores)
            prob = float(clf.predict_proba(np.array(features, dtype=np.float32).reshape(1, -1))[:, 1][0])
            decision_learned = "nsfw" if prob >= learned_threshold else "sfw"

            filename = os.path.basename(str(media_path))
            is_face = filename.startswith("Faces")

            if is_face:
                faces_total += 1
                if decision_rules == "sfw" and decision_learned == "sfw":
                    faces_sfw_rules += 1
            else:
                if decision_learned == "nsfw":
                    nsfw_hits_learned += 1

        failed = False
        if faces_total > 0 and faces_sfw_rules != faces_total:
            print(f"Sanity check FAILED: {faces_sfw_rules}/{faces_total} Faces_* SFW in both modes")
            failed = True
        if nsfw_hits_learned == 0:
            print("Sanity check FAILED: no non-Faces_* files detected as NSFW in learned mode")
            failed = True

        if failed:
            raise SystemExit(1)
        else:
            print("Sanity check PASSED")
            return

    model, preprocess, tokenizer, device = load_model(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        legacy_model=args.legacy_model,
    )
    text_features = encode_prompts(PROMPTS, model, tokenizer, device)

    media_files = list(iter_media_files(args.input_path))

    results = []

    learned_clf = None
    learned_threshold = 0.5
    classifier_path = Path(args.classifier_model_path)
    thresholds_path = classifier_path.with_name(
        classifier_path.stem + "_thresholds.json"
    )
    if args.mode == "learned" and Path(args.classifier_model_path).exists():
        learned_clf = joblib.load(args.classifier_model_path)
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                thresholds = json.load(f)
            learned_threshold = float(thresholds.get("learned_threshold", 0.5))

    for media_path in tqdm(media_files, desc="Scanning"):
        try:
            if media_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                scores = score_video_frames(
                    media_path,
                    model,
                    preprocess,
                    text_features,
                    device,
                    fps=args.fps,
                    batch_size=args.batch_size,
                )
            else:
                scores = score_image_path(
                    str(media_path),
                    model,
                    preprocess,
                    text_features,
                    device,
                    batch_size=args.batch_size,
                )

            if args.mode == "learned" and learned_clf is not None:
                features = build_features_from_scores(scores)
                prob = float(learned_clf.predict_proba([features])[0][1])
                decision = "nsfw" if prob >= learned_threshold else "sfw"
                reason = f"learned prob {prob:.3f} (thresh {learned_threshold:.3f})"
            else:
                decision, reason = _policy_decision(scores, {k: args.threshold for k in CONFIG})

            # Default v0.2 schema: explicit columns per prompt
            base_result = {
                "file": str(media_path),
                "safe_sim": float(scores[0]),
                "female_nipples_sim": float(scores[1]),
                "male_nipples_sim": float(scores[2]),
                "penis_sim": float(scores[3]),
                "vulva_sim": float(scores[4]),
                "anus_sim": float(scores[5]),
                "breast_sim": float(scores[6]),
                "chest_sim": float(scores[7]),
                "bikini_sim": float(scores[8]),
                "lingerie_sim": float(scores[9]),
                "cleavage_sim": float(scores[10]),
                "decision": decision,
                "reason": reason,
                "mode": args.mode,
                "model_name": args.model,
                "feature_version": FEATURE_VERSION,
            }

            if args.legacy_csv:
                # Add legacy-compatible columns
                base_result.update(
                    {
                        "nipples_sim": max(base_result["female_nipples_sim"], base_result["male_nipples_sim"]),
                        "clothing_sim": max(
                            base_result["bikini_sim"],
                            base_result["lingerie_sim"],
                            base_result["cleavage_sim"],
                        ),
                        "nsfw_youtube_guess": decision,
                    }
                )

            results.append(base_result)

            if args.verbose or args.debug:
                print(f"{media_path}: {decision} - {reason}")
        except Exception as exc:
            print(f"Error processing {media_path}: {exc}")

    if not results:
        print("No media files found to scan.")
        return

    # Preserve stable column ordering by using the keys from the first result
    fieldnames = list(results[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {args.out}")

if __name__ == '__main__':
    main()
