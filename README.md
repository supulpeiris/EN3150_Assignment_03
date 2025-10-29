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



Question 8: Which optimizer did you use for training, and why did you choose it?

Optimizer Used: Adam (Adaptive Moment Estimation)

Why we Chose Adam Optimizer:
1. Adaptive Learning Rate Capability
  Problem with SGD: Uses a single learning rate for all parameters throughout training
  Adam Solution: Automatically adjusts learning rates for each parameter individually
  Benefit: No need for manual learning rate scheduling, which is complex for beginners

2. Combines Best Features of Other Optimizers
  Momentum from SGD+Momentum: Helps accelerate convergence in relevant directions
  Adaptive learning rates from RMSProp: Adjusts learning rates based on recent gradient magnitudes
  Result: More robust and efficient than using either approach alone


4. Handles Sparse Gradients Well
  Our case: Image classification with 17 classes creates sparse gradient scenarios
  Adam's advantage: Adaptive learning rates prevent parameters from receiving insufficient updates

5. Fast Convergence in Practice
  Empirical evidence: Adam typically converges faster than SGD in deep learning applications
  Our results: Achieved reasonable accuracy (63% training, 48% validation) in just 20 epochs

6. Default Choice in Modern Deep Learning
  Industry standard: Most research papers and practical applications use Adam for CNN training
  Well-tuned defaults: β1=0.9, β2=0.999, ε=1e-7 work well across diverse problems

7. Comparison with Alternatives:
  Standard SGD: Too slow, requires careful learning rate tuning
  SGD with Momentum: Better than SGD but still needs manual learning rate adjustments
  RMSProp: Good but doesn't incorporate momentum concept

Experimental Verification:
  In our optimizer comparison, Adam typically outperforms both standard SGD and SGD with Momentum in terms of:
  Final validation accuracy
  Training stability
  Convergence speed

Question 9: How do you select the learning rate? [10 marks]
Learning Rate Selection Strategy:
1. Start with Well-Established Defaults
python
# For Adam optimizer, the default learning rate is:
learning_rate = 0.001
Rationale: 0.001 is a well-researched starting point that works across many domains
Empirical success: Proven effective in numerous research papers and practical applications

2. Consider the Optimizer Type
Adam: Default 0.001 works well due to adaptive learning rates
SGD: Typically requires higher learning rates (0.01-0.1)
RMSProp: Similar to Adam, around 0.001

3. Dataset-Specific Considerations
  Our Jute Pest Dataset: 7,235 images, 17 classes
  Moderate complexity: Not too simple, not extremely complex
  Learning rate 0.001: Appropriate for this scale of problem

B. Cyclical Learning Rates
  Start with low LR, gradually increase, then decrease
  Helps find optimal learning rate range

6. Monitoring and Adjustment During Training
    Signs of Good Learning Rate:
      Training loss decreases steadily
      Validation loss follows similar pattern
      Reasonable convergence within expected epochs
    
    Signs of Learning Rate Too High:
      Loss becomes NaN or extremely large
      Training is unstable with large oscillations
      Model doesn't converge
    
    Signs of Learning Rate Too Low:
      Very slow convergence
      Training gets stuck in suboptimal solutions
      Many epochs needed for minimal improvement

# Step 3: Monitor training behavior
# - If converging too slowly → consider increasing LR
# - If unstable → decrease LR
# - Our case: 0.001 showed reasonable convergence
8. Validation-Based Adjustment
Use validation loss as guide: If validation loss plateaus or increases, reduce learning rate

Early stopping: Prevent overfitting while maintaining optimal learning dynamics

9. Practical Considerations for This Assignment:
Time constraints: Limited epochs (20) favor slightly higher learning rates
Dataset size: 7,235 images can support stable training with 0.001
Model complexity: Our 3-layer CNN is moderate, not extremely deep
Final Learning Rate Justification:
We used learning_rate = 0.001 because:

Well-established default for Adam optimizer
Appropriate for our dataset size and complexity
Showed stable convergence in our training curves
Supported by adaptive scheduling that automatically adjusts if needed
Balances convergence speed with training stability


SGD:
  Training Accuracy:   0.0531
  Validation Accuracy: 0.0726
  Training Loss:       nan
  Validation Loss:     nan
  Overfitting Gap:     -0.0196

SGD+Momentum:
  Training Accuracy:   0.0531
  Validation Accuracy: 0.0726
  Training Loss:       nan
  Validation Loss:     nan
  Overfitting Gap:     -0.0196

Adam:
  Training Accuracy:   0.2918
  Validation Accuracy: 0.2421
  Training Loss:       2.2507
  Validation Loss:     2.7463
  Overfitting Gap:     0.0497


Question 10: Optimizer Comparison

"The optimizer comparison revealed significant differences in performance:
SGD and SGD+Momentum failed completely, achieving only 5.31% accuracy (essentially random guessing) and exhibiting NaN loss values indicating numerical instability due to exploding gradients.
Adam optimizer performed substantially better, achieving 29.18% training accuracy and 24.21% validation accuracy with stable loss values.
This demonstrates that Adam's adaptive learning rates are crucial for stable training of our CNN architecture on the Jute Pest dataset."

Question 11: Momentum Parameter Impact

"The momentum parameter analysis showed that momentum alone cannot compensate for inappropriate learning rates. Both SGD (momentum=0) and SGD+Momentum (momentum=0.9) failed identically, indicating that the fundamental issue was the fixed learning rate being too high.
Momentum's role is to accelerate convergence in the right direction, but it requires a properly tuned learning rate to be effective. When the learning rate is too high, momentum actually worsens the instability by amplifying gradient oscillations."


