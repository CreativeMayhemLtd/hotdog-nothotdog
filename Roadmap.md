# Roadmap to V1.0

This document outlines the planned improvements for Hotdog_NotHotdog from V0.1.0 to V1.0, making it a production-ready, high-accuracy NSFW detector.

## Core Enhancements
- **Model Improvements**: Experiment with CLIP ViT-L/14 or fine-tuned variants for better generalization. Add support for multiple CLIP models.
- **Feature Engineering**: Incorporate more prompts (e.g., contextual NSFW like "graphic violence") or use raw embeddings for deeper learning.
- **Accuracy Boost**: Implement ensemble methods (combine RandomForest with SVM) and cross-validation for more robust training.
- **Performance**: Optimize for GPU/CPU, add batch processing, and reduce memory usage for large datasets.

## User Experience
- **CLI Polish**: Add progress bars for training, better error messages, and config files for custom thresholds/prompts.
- **Output Formats**: Support JSON, XML, or database exports beyond CSV.
- **GUI/Web Interface**: Optional web app for drag-and-drop scanning.
- **Integration Hooks and ComfyUI Module**: Add APIs/hooks for seamless integration with tools like ComfyUI for real-time NSFW filtering in workflows.

## Robustness & Testing
- **Edge Cases**: Handle corrupted files, large videos, and mixed formats better.
- **Testing Suite**: Add unit tests (pytest) for functions, integration tests for full pipelines.
- **Benchmarking**: Compare against other NSFW detectors (e.g., NudeNet) on public datasets.

## Deployment & Scaling
- **Packaging**: Create a PyPI package or Docker image for easy install.
- **API Mode**: Add a REST API for integration with other tools.
- **Multi-Platform**: Ensure macOS compatibility and test on various hardware.

## Ethical & Legal
- **Bias Mitigation**: Audit for biases in training data and add disclaimers.
- **Privacy**: Ensure no data is sent externally; all local processing.
- **Licensing**: Finalize MIT or similar, with clear usage guidelines.

## Milestones
- **V0.2**: Implement ensemble models and basic testing.
- **V0.5**: Add API and improved CLI.
- **V1.0**: Full benchmarking, packaging, and ethical review.

Contributions and feedback welcome!