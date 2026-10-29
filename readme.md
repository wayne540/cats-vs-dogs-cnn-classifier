# Cats vs Dogs CNN Classifier 🐱🐶

A deep learning project that uses a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras to classify images as either cats or dogs.

The project explores the complete image classification workflow, from dataset preparation and image preprocessing to CNN development, model training, performance evaluation, and predictions on previously unseen images.

The notebook also supports **batch prediction**, allowing multiple uploaded images to be classified in a single model inference call.

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Machine Learning Workflow](#machine-learning-workflow)
* [CNN Architecture](#cnn-architecture)
* [Data Preprocessing](#data-preprocessing)
* [Model Training](#model-training)
* [Model Evaluation](#model-evaluation)
* [Making Predictions](#making-predictions)
* [Installation and Setup](#installation-and-setup)
* [Running the Project](#running-the-project)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Key Learning Outcomes](#key-learning-outcomes)
* [Author](#author)

---

## Project Overview

Image classification is a computer vision task in which a machine learning model learns to assign images to predefined categories.

In this project, a CNN learns to distinguish between images of cats and dogs by identifying useful visual patterns in the training dataset.

Rather than relying on manually defined image features, the CNN learns hierarchical representations directly from image pixels. Early convolutional layers can learn simple patterns, while deeper layers can combine those patterns into more complex visual features.

The trained model produces a score for the Dog class. This score is used to classify an image as either a cat or a dog.

The project is implemented in Python using TensorFlow and Keras, with NumPy, Matplotlib, and Pillow supporting data processing and visualization.

## Objectives

The main objectives of this project are to:

1. Understand the fundamental principles of CNN-based image classification.
2. Develop an image preprocessing pipeline for preparing raw images for deep learning.
3. Build a CNN architecture using TensorFlow and Keras.
4. Apply data augmentation to introduce variation into the training images.
5. Train a binary image classifier using binary cross-entropy loss.
6. Monitor training performance using validation metrics and early stopping.
7. Visualize training progress and classification results.
8. Evaluate the model using metrics beyond accuracy.
9. Implement single-image and batch-image prediction.
10. Understand the practical challenges associated with training and evaluating deep learning models.

## Features

* **Image preprocessing:** Converts supported image formats to RGB, resizes images, and normalizes pixel values.
* **Dataset splitting:** Separates data into training, validation, and testing subsets.
* **Data augmentation:** Applies random horizontal flipping and translation during training.
* **CNN implementation:** Uses convolutional layers to learn visual features from images.
* **Binary classification:** Distinguishes between cats and dogs using a sigmoid output layer.
* **Early stopping:** Monitors validation loss and restores the best-performing model weights.
* **Performance monitoring:** Tracks accuracy, precision, AUC, and loss.
* **Visualization:** Displays sample images, class distributions, training history, and predictions.
* **Batch inference:** Classifies multiple uploaded images in a single model prediction call.
* **Model persistence:** Saves the trained model for later use.

## Technologies Used

| Technology       | Purpose                                                   |
| ---------------- | --------------------------------------------------------- |
| Python           | Main programming language                                 |
| TensorFlow       | Deep learning framework                                   |
| Keras            | CNN construction, training, and inference                 |
| NumPy            | Numerical operations and image array manipulation         |
| Matplotlib       | Data visualization and prediction plots                   |
| Pillow           | Image loading, RGB conversion, and resizing               |
| Scikit-learn     | Dataset splitting and classification evaluation           |
| Jupyter Notebook | Interactive development and experimentation               |
| Google Colab     | Optional cloud-based notebook execution and image uploads |
| Git and GitHub   | Version control and project hosting                       |

## Dataset

This project uses the Microsoft Cats vs Dogs image dataset for binary image classification.

**Download:** [Microsoft Cats vs Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765&msockid=04d2e6ff2dbe6ac11da3f0b82c1e6b9b&utm_source=chatgpt.com)

The dataset is provided as a ZIP archive containing images organized into `Cat` and `Dog` directories.

### Setup

1. Download the dataset from the official Microsoft page.
2. Extract the ZIP archive.
3. Update the dataset path in the notebook if necessary.
4. Run the preprocessing cells to validate the images and prepare the training, validation, and test datasets.

**Note:** The archive is approximately 787 MB. Some images may be corrupted and should be checked before training.


### Dataset preparation

Before training, the notebook:

1. Locates the image files.
2. Associates each image with its class label.
3. Checks image validity.
4. Splits the valid image paths and labels into training, validation, and testing subsets.
5. Loads and preprocesses images when the TensorFlow datasets are iterated over.

The dataset is not included in this repository because of its size.

**Dataset note:** Record the original dataset source and applicable licence when publishing the project. If using Microsoft's Cats vs Dogs dataset, consult the dataset's official documentation for its current availability and terms.

## Project Structure

The repository is organized as follows:

```text
cats-vs-dogs-cnn-classifier/
│
├── notebooks/
│   └── cats_vs_dogs_cnn.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

### File descriptions

| File                               | Description                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `notebooks/cats_vs_dogs_cnn.ipynb` | Main notebook containing data preparation, CNN implementation, training, evaluation, and prediction. |
| `README.md`                        | Project documentation.                                                                               |
| `requirements.txt`                 | Python dependencies required to run the project.                                                     |
| `.gitignore`                       | Specifies files and directories that Git should ignore.                                              |

The notebook filename can be adjusted to match the actual filename in the repository.

## Machine Learning Workflow

The project follows a supervised deep learning workflow.

### 1. Dataset preparation

Image paths and their associated labels are collected and checked before being passed into the training pipeline.

The labels are encoded as follows:

| Class | Label |
| ----- | ----: |
| Cat   |     0 |
| Dog   |     1 |

Correct labelling is essential because the model learns from the relationship between each image and its assigned target.

### 2. Dataset splitting

The dataset is divided into three subsets:

| Subset     | Percentage | Purpose                                                                     |
| ---------- | ---------: | --------------------------------------------------------------------------- |
| Training   |        70% | Used to learn model parameters.                                             |
| Validation |        15% | Used to monitor generalization during training and support model selection. |
| Testing    |        15% | Used for final evaluation on held-out data.                                 |

Stratified splitting helps preserve the class proportions across the subsets.

The test dataset should remain separate from model development decisions to reduce the risk of test-data leakage.

### 3. Image preprocessing

Images are converted to RGB, resized to `128 × 128` pixels, and normalized to the range `[0, 1]`.

This ensures that images entering the CNN have consistent dimensions and numerical scales.

### 4. Data augmentation

The training pipeline applies random transformations, including:

* Horizontal flipping.
* Small horizontal and vertical translations.

These transformations introduce variation into training images and can help the model generalize to different image orientations and positions.

Augmentation is applied during training rather than used to alter the validation and test datasets.

### 5. Model construction

A CNN is built using convolutional layers, max-pooling operations, global average pooling, dropout, and a sigmoid output layer.

### 6. Model training

The model learns from the training dataset using the Adam optimizer and binary cross-entropy loss.

Validation performance is monitored after each epoch.

Early stopping terminates training when validation loss stops improving for the configured patience period.

### 7. Evaluation and prediction

The trained model is evaluated using the held-out test dataset. It can then classify previously unseen images uploaded by the user.

## CNN Architecture

The model uses a sequential feature-extraction pipeline followed by a binary classification head.

| Layer                  | Configuration               | Purpose                                      |
| ---------------------- | --------------------------- | -------------------------------------------- |
| Input                  | `128 × 128 × 3`             | Receives RGB images.                         |
| Data augmentation      | Random flip and translation | Introduces image variation during training.  |
| Convolution            | 64 filters, `3 × 3` kernel  | Learns early visual features.                |
| Max pooling            | Default `2 × 2` pooling     | Reduces spatial dimensions.                  |
| Convolution            | 128 filters, `3 × 3` kernel | Learns more complex features.                |
| Max pooling            | Default `2 × 2` pooling     | Further reduces spatial dimensions.          |
| Convolution            | 256 filters, `3 × 3` kernel | Learns higher-level visual representations.  |
| Max pooling            | Default `2 × 2` pooling     | Reduces spatial dimensions again.            |
| Global average pooling | Spatial averaging           | Converts feature maps into a feature vector. |
| Dropout                | Rate `0.3`                  | Regularizes the model during training.       |
| Dense output           | 1 neuron, sigmoid           | Produces the Dog-class score.                |

### Why use a CNN?

Images contain spatial relationships that conventional fully connected networks do not naturally exploit as efficiently.

Convolutional layers use learnable filters to detect local patterns, while pooling operations reduce spatial dimensions.

As the network deepens, its learned feature representations can become increasingly useful for distinguishing between the two classes.

### Why use global average pooling?

Global average pooling averages each feature map into a single value.

Compared with flattening the entire final feature map, this reduces the number of parameters entering the classification head and can help limit model complexity.

### Why use dropout?

Dropout randomly deactivates a fraction of the layer's activations during training.

This encourages the model to learn less dependent representations and may reduce overfitting.

## Data Preprocessing

The preprocessing pipeline performs the following operations:

1. Reads an image file.
2. Decodes the image.
3. Converts it to three-channel RGB.
4. Resizes it to `128 × 128`.
5. Converts pixel values to floating-point numbers.
6. Normalizes pixel values by dividing by 255.
7. Converts labels to integer tensors.

The resulting image tensor has the shape:

```text
(128, 128, 3)
```

When images are grouped into a batch of 128, the expected image tensor shape is:

```text
(128, 128, 128, 3)
```

The first dimension represents the batch size.

The preprocessing implementation uses Pillow where necessary to convert unusual image modes into RGB before TensorFlow receives the resulting image arrays.

## Model Training

The model is compiled using the following configuration:

| Parameter               | Configuration        |
| ----------------------- | -------------------- |
| Optimizer               | Adam                 |
| Initial learning rate   | `0.001`              |
| Loss function           | Binary cross-entropy |
| Accuracy metric         | Binary accuracy      |
| Additional metrics      | Precision and AUC    |
| Maximum epochs          | 20                   |
| Batch size              | 128                  |
| Early stopping monitor  | Validation loss      |
| Early stopping patience | 3 epochs             |
| Restore best weights    | Enabled              |

These are the configured training settings, not a claim about the model's achieved performance.

### Binary cross-entropy

Binary cross-entropy measures the discrepancy between the predicted class scores and the true binary labels.

Minimizing this loss encourages the model to assign higher scores to the correct class.

### Adam optimizer

Adam adjusts the model's trainable parameters using estimates of gradients and their historical magnitudes.

The learning rate controls the scale of the optimizer's parameter updates.

### Early stopping

Early stopping monitors validation loss and interrupts training when improvement stalls.

Restoring the best weights helps retain the model state associated with the lowest observed validation loss rather than automatically using the final epoch.

## Model Evaluation

The notebook tracks accuracy, precision, AUC, and loss during training.

The held-out test dataset should be used to assess final model performance after development decisions have been completed.

### Evaluation metrics

**Accuracy**

Measures the fraction of predictions classified correctly at the selected classification threshold.

**Precision**

Measures the proportion of images predicted as dogs that actually belong to the Dog class.

**Area Under the ROC Curve (AUC)**

Measures how effectively the model ranks positive examples above negative examples across different classification thresholds.

**Loss**

Measures prediction error according to the binary cross-entropy objective.

Additional evaluation can include recall, F1-score, and a confusion matrix to provide a more complete picture of classification performance.

Actual evaluation results should be recorded after the model has been trained and tested.

## Making Predictions

The project supports predictions on images uploaded by the user.

### Single-image prediction

A user can upload an image, preprocess it, and pass it to the trained CNN.

The model returns a score for the Dog class.

For example, a hypothetical output of `0.85` corresponds to:

| Class | Model score |
| ----- | ----------: |
| Dog   |         85% |
| Cat   |         15% |

With a classification threshold of `0.5`, the predicted class is Dog.

### Batch prediction

The notebook also supports classifying multiple images in one inference call.

For example, four uploaded images can be:

1. Converted to RGB.
2. Resized to the model's expected dimensions.
3. Normalized.
4. Combined into a batch.
5. Passed through the CNN together.
6. Displayed with their predicted classes and class scores.

Batch inference avoids making a separate model prediction call for every image.

### Important distinction about confidence

The sigmoid output is a model score for the Dog class. The complementary score is calculated as `1 - dog_score` for the Cat class.

These scores should not automatically be interpreted as calibrated probabilities of correctness. A model can produce a high score even when it misclassifies an unfamiliar image.

## Installation and Setup

### Requirements

The project uses Python and several machine learning and image-processing libraries.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

A suitable starting point for `requirements.txt` is:

```text
tensorflow
numpy
matplotlib
Pillow
scikit-learn
jupyter
```

For reproducible experiments, use a tested Python environment and pin the dependency versions after confirming that the project runs successfully.

### Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/cats-vs-dogs-cnn-classifier.git
```

Replace `YOUR-USERNAME` with your GitHub username.

Move into the project directory:

```bash
cd cats-vs-dogs-cnn-classifier
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Prepare the dataset

Obtain the dataset from its documented source and organize the images according to the directory structure expected by the notebook.

Update the dataset paths in the notebook if your local directory structure differs from the original configuration.

The dataset and trained model are not included in the repository by default.

## Running the Project

1. Clone the repository.
2. Install the dependencies.
3. Download and prepare the dataset.
4. Open the notebook.
5. Run the preprocessing and dataset-splitting cells.
6. Create the training, validation, and test datasets.
7. Build and compile the CNN.
8. Train the model.
9. Review the training and validation curves.
10. Evaluate the final model using the test dataset.
11. Upload images and run the prediction cells.

The notebook can be run locally in Jupyter or VS Code, or adapted for Google Colab.

**Environment note:** The image-upload code using `google.colab.files.upload()` is specific to Google Colab. A local Jupyter or VS Code notebook needs a different method for selecting image files.

## Limitations

Several limitations should be considered when interpreting the results.

* **Dataset dependence:** The model's performance depends on the quality, diversity, and labelling of the training images.
* **Generalization:** Performance on the test dataset does not guarantee equivalent performance on every real-world image.
* **Image quality:** Blurry, poorly lit, or heavily obstructed images may be harder to classify.
* **Class scope:** The model is trained to distinguish cats from dogs. It is not designed to identify arbitrary objects or reliably reject images containing neither class.
* **Prediction scores:** High sigmoid scores do not guarantee that a prediction is correct.
* **Computational requirements:** Training time depends on dataset size, hardware, and TensorFlow configuration.
* **Input decoding:** Unusual or corrupted image files may require additional validation or conversion.

The classifier should therefore be treated as a two-class image classification model rather than a general-purpose object recognition system.

## Future Improvements

Potential extensions include:

* Experimenting with deeper CNN architectures.
* Comparing the baseline CNN with transfer-learning models such as MobileNetV2 or EfficientNet.
* Tuning the learning rate, batch size, dropout rate, and number of convolutional filters.
* Adding a learning-rate scheduler.
* Evaluating performance using precision, recall, F1-score, ROC curves, and confusion matrices.
* Investigating class imbalance and its effect on predictions.
* Adding an explicit mechanism to flag uncertain predictions.
* Building a simple web interface for image uploads.
* Improving input-pipeline performance.
* Tracking experiments and saving model evaluation results.
* Testing the model on images collected from sources outside the training dataset.

These improvements would help assess how well the model generalizes and how it might be developed into a more practical computer vision application.

## Key Learning Outcomes

This project provides practical experience with:

* Supervised deep learning.
* CNN architecture design.
* Image loading and preprocessing.
* TensorFlow data pipelines.
* Data augmentation.
* Binary cross-entropy optimization.
* Validation-based model selection.
* Early stopping and regularization.
* Model evaluation.
* Batch inference.
* Saving and loading trained models.
* Identifying and handling image-decoding errors.

The primary objective is not simply to obtain a high accuracy score. It is to understand how the complete image classification pipeline works and how each design decision influences the model's behaviour.

## Author

**Omanyo Wayne**

Electrical and Electronics Engineering student with an interest in machine learning, deep learning, and practical AI development.

This project is part of my continued exploration of deep learning through implementation, experimentation, and analysis.

---

*Built with Python, TensorFlow, and Keras.*

