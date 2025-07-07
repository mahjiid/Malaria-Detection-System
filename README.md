Focuses on building and comparing two deep learning models for classifying malaria cell images as "Parasitized" or "Uninfected" using a Convolutional Neural Network (CNN) and transfer learning with MobileNetV2. Below, I’ll summarize the key components, analyze the results, and address potential issues or improvements, as requested.



### Summary of the Notebook

## Objective
The notebook aims to classify malaria cell images into "Parasitized" (infected) and "Uninfected" (healthy) categories using two approaches:
1. A custom CNN model.
2. A transfer learning model based on MobileNetV2.
It compares their performance based on validation accuracy and test results.

# Steps and Implementation
1. Libraries Imported:
   - os, cv2, numpy, tqdm, matplotlib.pyplot: For file handling, image processing, numerical operations, progress tracking, and visualization.
   - tensorflow, keras: For building and training deep learning models.
   - sklearn.model_selection.train_test_split: For splitting data into train and test sets.

2. Data Loading (load_cell_images):
   - Loads images from `Parasitized` and `Uninfected` subfolders in the specified directory (`C:/Users/wagmiman/Python Projects/cell_images`).
   - Resizes images to 64x64 pixels.
   - Labels: Parasitized = 1, Uninfected = 0.
   - Converts images and labels to NumPy arrays.
   - Output: X (shape: (27558, 64, 64, 3)), y (shape: (27558,)).
   - Normalizes pixel values to [0, 1] by dividing by 255.0.
   - Note: The provided code uses a placeholder load_cell_images function with dummy data (n_samples=1000), likely for testing, which affects the results.

3. Data Preprocessing:
   - Splits data into training (80%) and testing (20%) sets using train_test_split with stratification to maintain class balance.

4. Custom CNN Model (cnn_model):
   - Architecture:
     - Conv2D (16 filters, 3x3, ReLU) → BatchNormalization → MaxPooling2D (2x2)
     - Conv2D (32 filters, 3x3, ReLU) → BatchNormalization → MaxPooling2D (2x2)
     - Flatten → Dense (32, ReLU) → Dropout (0.5) → Dense (1, sigmoid)
   - Optimizer: Adam
   - Loss: Binary cross-entropy
   - Metrics: Accuracy
   - Note: This model is defined but not used in the provided code; only MobileNetV2 is trained.

5. Transfer Learning Model (MobileNetV2):
   - Uses pre-trained MobileNetV2 (weights from ImageNet, excluding top layers).
   - Architecture:
     - Base model (MobileNetV2, frozen layers)
     - GlobalAveragePooling2D → Dropout (0.5) → Dense (32, ReLU) → Dense (1, sigmoid)
   - Optimizer: Adam
   - Loss: Binary cross-entropy
   - Metrics: Accuracy
   - Early stopping: Monitors validation loss, patience=3, restores best weights.

6. Training:
   - Trains MobileNetV2 model on training data for up to 10 epochs with a batch size of 32.
   - Uses early stopping to prevent overfitting.
   - Plots training and validation accuracy over epochs.

7. Evaluation:
   - Evaluates MobileNetV2 on the test set.
   - Results: Test Loss = 0.6932, Test Accuracy = 0.5000 (based on dummy data).
   - Predicts on 5 test samples, showing predicted vs. true labels ( Sample 1: Predicted=Uninfected, True=Parasitized).

8. Results Summary:
   - The notebook claims both models (CNN and MobileNetV2) yield similar validation accuracies.
   - Custom CNN: Validation accuracy remains nearly constant after the first epoch, with gradual improvement in training accuracy.
   - MobileNetV2: Validation accuracy fluctuates but training accuracy increases steadily.
   - Note: The results are based on dummy data (n_samples=1000`, random images/labels), leading to poor performance (50% accuracy, equivalent to random guessing).
