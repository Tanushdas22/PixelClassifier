# PixelClassifier

## Objective
This project implements a simple **LeNet-style Convolutional Neural Network (CNN)** to classify handwritten digits from the MNIST dataset. The notebook performs:

- 5-fold Stratified Cross-Validation with fixed hyperparameters.
- Selection of the best hyperparameter combination.
- Retraining of the best model on the full training set.
- Evaluation of test accuracy.

---

## Dataset
- **MNIST**: 70,000 grayscale images of handwritten digits (0–9), 28×28 pixels.
- **Split**: 60,000 training images, 10,000 test images.
- **Preprocessing**:
  - Pixel values normalized to [0, 1].
  - Images reshaped to include a single channel: `(28, 28, 1)`.
  - Labels one-hot encoded during training for categorical cross-entropy.

---

## CNN Model Architecture
The model is a compact **LeNet-style CNN**:

1. **Input** → Conv2D (ReLU) → MaxPooling2D  
2. Conv2D (ReLU) → MaxPooling2D  
3. Flatten → Dense (128 units, ReLU) → Dense (10 units, Softmax)

**Hyperparameters**:

- Filters: 16 or 32
- Dense layer: 128 units
- Kernel size: (3,3)
- Optimizer: Adam
- Learning rate: 0.001 or 0.01
- Epochs per fold: 3
- Batch size: 128

---

## 5-Fold Stratified Cross-Validation
- Each hyperparameter combination is evaluated with 5-fold StratifiedKFold.
- Accuracy is recorded per fold.
- Mean validation accuracy determines the best combination.

**Hyperparameter combinations tested:**

| Filters | Learning Rate |
|---------|---------------|
| 16      | 0.001         |
| 16      | 0.01          |
| 32      | 0.001         |
| 32      | 0.01          |

**Mean accuracies per combination:**

| Filters | Learning Rate | Mean Accuracy |
|---------|---------------|---------------|
| 16      | 0.001         | 0.9821        |
| 16      | 0.01          | 0.9819        |
| 32      | 0.001         | 0.9850        |
| 32      | 0.01          | 0.9815        |

**Best combination**: `filters = 32, learning rate = 0.001`

---

## Retraining on Full Dataset
- The best model is retrained on all 60,000 training images for 3 epochs.
- Final test accuracy on 10,000 test images: **98.68%**.

---

## Results & Discussion
- The compact LeNet-style CNN achieves **high accuracy** despite only two convolutional layers and 3 training epochs.
- Comparatively deeper models can reach 99.2–99.4% with more epochs and regularization.
- Potential improvements:
  - Add another Conv-Pool block.
  - Increase training epochs.
  - Apply data augmentation.
  - Use batch normalization or dropout.

---

## References
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of IEEE, 86(11), 2278–2324.  
2. [Keras Conv2D and Convolutional Layers](https://pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/)  
3. [Convolutional Neural Network (CNN) Overview](https://blog.gopenai.com/convolutional-neural-network-cnn-054ac70d40ec)  

---

## Usage
```bash
# Clone repo
git clone <repo_url>
cd <repo_dir>

# Open notebook in Jupyter or Colab
jupyter notebook Lab3_LeNetCNN.ipynb
# or upload to Google Colab
