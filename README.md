# Braille to English Translator

## Overview
This project implements a Convolutional Neural Network (CNN) to translate Braille characters into English letters (a-z). The model is trained on a dataset of Braille images and achieves high accuracy in recognizing Braille patterns.

## Features
- **Dataset**: Utilizes the [Braille Character Dataset](https://www.kaggle.com/datasets/shanks0465/braille-character-dataset) from Kaggle.
- **Preprocessing**: Images are resized to 28x28 pixels and normalized for model input.
- **Model**: CNN with multiple convolutional layers, max pooling, dropout, and dense layers for classification.
- **Training**: Trained for 30 epochs with a validation split of 20%.
- **Evaluation**: Achieves ~90% validation accuracy.
- **Prediction**: Translates individual Braille images to corresponding English letters.

## Requirements
- Python 3.x
- Libraries:
  ```bash
  pip install tensorflow opencv-python numpy matplotlib scikit-learn kagglehub

## Model Architecture
- Input: 28x28x1 grayscale images.
- Layers:
  - Conv2D (32 filters, 3x3, ReLU) → MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3, ReLU) → MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3, ReLU)
  - Flatten
  - Dense (128 units, ReLU) → Dropout (0.4)
  - Dense (26 units, softmax)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

Results
- Training Accuracy: ~93-94%
- Validation Accuracy: ~90-91%

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset provided by shanks0465 on Kaggle.
- Built using TensorFlow, OpenCV, and other open-source libraries.
