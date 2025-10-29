# EN3150_Assignment_03
EN3150 Assignment 03: Simple convolutional neural network  to perform classification

Phase 1: Setup and Data Preparation
Set Up Your Environment:

Create a new Python environment (using conda or venv) to manage your packages.
Install essential libraries: tensorflow, numpy, pandas, matplotlib, scikit-learn, seaborn, opencv-python (for image handling if needed).
Set up a GitHub repository. Commit your initial empty project and this environment setup.

Prepare Your Dataset (Jute Pest):
Download the dataset from the UCI repository.
Understand the Data: How many classes (types of pests) are there? What is the image size? How many images are there per class? This will influence your model design.
Load and Preprocess the Images:
Read all images.
Resize them to a consistent size (e.g., 128x128 or 224x224 pixels). A smaller size will make training faster.
Normalize the pixel values from [0, 255] to [0, 1] or [-1, 1]. This helps the model converge faster.
Split the Data: Use train_test_split from sklearn twice to get:

70% Training: For the model to learn.

15% Validation: For tuning hyperparameters and checking for overfitting during training.

15% Test: For the final, unbiased evaluation of your model's performance. Do not touch this set until the very end.

Phase 2: Build, Train, and Evaluate Your Custom CNN
Build the CNN Model (Q4 & Q5):

Follow the basic architecture given in the assignment.

You need to choose the values for:

x1, x2: Number of filters (e.g., 32, 64). Start with smaller numbers.

m1, m2: Kernel sizes (e.g., 3x3 is standard).

x3: Number of units in the Dense layer (e.g., 128, 64).

d: Dropout rate (e.g., 0.5 is common).

K: Number of units in the output layer (this is determined by your dataset - the number of pest classes).

Example Code Skeleton:

python
from tensorflow.keras import models, layers

model = models.Sequential()
# Conv Block 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
# Conv Block 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Classifier
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax')) # num_classes from your dataset
Commit your model code to GitHub.

Justify Your Choices (Q6):

Activation Functions:

For hidden layers (Conv, Dense), ReLU is the standard choice. It's simple, avoids the vanishing gradient problem, and helps models learn faster.

For the output layer, Softmax is used because it converts the outputs into probabilities for each class, which is perfect for multi-class classification.

Train the Model (Q7, Q8, Q9):

Compile the Model: Choose an optimizer, loss function, and metrics.

Optimizer (Q8): A good starting point is Adam. It's adaptive and often works well without much tuning. You can justify it by saying it combines the advantages of other methods like AdaGrad and RMSProp.

Loss Function: sparse_categorical_crossentropy (if your labels are integers) or categorical_crossentropy (if your labels are one-hot encoded).

Metrics: accuracy.

Learning Rate (Q9): Start with the default learning rate of the optimizer (e.g., Adam's default is 0.001). You can say you used the default as it's a well-tested starting point. For a more advanced approach, you could use a learning rate scheduler (like reducing it on plateau).

Fit the Model: Train for 20 epochs, providing the training and validation data. This will return a history object containing the loss and accuracy for each epoch.

Plot the Training/Validation Loss: Use matplotlib to plot the loss over epochs. This visualizes if your model is learning and if it's overfitting (training loss keeps decreasing but validation loss starts increasing).

Optimizer Comparison (Q10 & Q11):

Create two new copies of your initial model.

Train one with SGD (standard) and another with SGD with Momentum.

Compare them with your original optimizer (e.g., Adam).

Metrics for Comparison:

Final Validation Accuracy: Which one performs best?

Training/Validation Loss Curves: Which one converges faster and more smoothly? SGD is often slower and may get stuck, while Momentum helps it converge faster. Adam is usually the fastest and most stable.

Impact of Momentum (Q11): Discuss how momentum helps the optimizer avoid local minima and navigate the loss landscape more effectively, leading to faster and more stable convergence.

Evaluate the Final Custom Model (Q12):

Use your test set (the 15% you held out) on your best custom model.

Calculate and report:

Test Accuracy

Confusion Matrix: Shows which classes are being confused with each other.

Precision and Recall: For each class. Precision = (True Positives) / (All predicted positives). Recall = (True Positives) / (All actual positives). These are crucial for imbalanced datasets.

Phase 3: Transfer Learning with State-of-the-Art Models
Choose and Fine-Tune Pre-trained Models (Q13, Q14, Q15):

Choose two models: Good choices are VGG16 (simpler) and ResNet50 (more complex, solves vanishing gradient problem). You can load them pre-trained on ImageNet directly from Keras.

Fine-Tuning Strategy:

Remove the original top classification layer.

Add your own new classification head (similar to your custom model: GlobalAveragePooling2D -> Dense -> Dropout -> Output).

You can choose to freeze the pre-trained layers initially and only train the new head, or do a few rounds of training with all layers unfrozen (this is more computationally expensive).

Train them on your same training/validation split.

Evaluate and Compare (Q16, Q17, Q18, Q19):

Plot the loss curves for the fine-tuned models.

Evaluate on your test set and record the same metrics (accuracy, precision, recall).

Compare (Q18): Create a table comparing your custom CNN, Fine-tuned VGG, and Fine-tuned ResNet on all metrics.

Discuss Trade-offs (Q19):

Custom CNN: Pros - Simple, fast to train, less computational power, easier to understand. Cons - Lower accuracy, requires good design skill.

Pre-trained Model: Pros - Very high accuracy, leverages learned features from a huge dataset, saves training time. Cons - "Black box", computationally heavy, can be overkill for simple problems, requires understanding of transfer learning.
