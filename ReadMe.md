# Human Emotion Detection

A comprehensive deep learning project for human emotion detection from images, implemented in a Jupyter notebook environment. The project explores multiple model architectures and optimization techniques to achieve efficient emotion classification.

## 🌟 Features

- Multiple Model Implementations:
  - LeNet-based CNN
  - ResNet34 (Custom Implementation)
  - Vision Transformer (ViT)
  - EfficientNetB4 (Transfer Learning)
  - MobileNetV2 (Transfer Learning)
  - Hugging Face ViT integration

- Advanced Training Techniques:
  - Data augmentation with random rotation, flipping, and contrast
  - Class weight balancing
  - Cutmix augmentation
  - Experiment tracking with Weights & Biases

- Model Optimization:
  - Dynamic quantization
  - TFLite conversion
  - ONNX export and optimization
  - Model pruning
  - TFRecord data pipeline

- Visualization Tools:
  - Feature map visualization
  - GradCAM implementation
  - Confusion matrix analysis
  - Training metrics visualization

## 🔢 Dataset

The project uses a dataset with three emotion classes:
- Angry
- Happy
- Sad

## 🛠️ Technical Details

- Framework: TensorFlow 2.x
- Additional Libraries:
  - OpenCV
  - Weights & Biases
  - Albumentations
  - ONNX Runtime
  - TFLite
  - Transformers (Hugging Face)

## 📊 Performance Comparison

| Model | Accuracy | Inference Time (GPU) |
|-------|----------|---------------------|
| EfficientNet | 84% | 0.15s |
| ONNX Runtime | 84% | 0.025s |
| TFLite Quantized | 84% | 0.3s |

## 📓 Notebook Structure

1. Data Management
   - Dataset downloading
   - Data preprocessing
   - Augmentation pipeline

2. Model Implementations
   - Multiple architecture implementations
   - Custom layers and blocks
   - Transfer learning setup

3. Training Pipeline
   - Loss functions and metrics
   - Training callbacks
   - Experiment tracking

4. Optimization Techniques
   - Model quantization
   - ONNX conversion
   - Pruning implementation

5. Visualization
   - Feature maps
   - Attention visualization
   - Performance metrics

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

