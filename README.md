## Project: Handwritten Digit Recognition using Deep Learning Convolutional Neural Network(CNN) 
```
project uses a Convolutional Neural Network to classify handwritten digits
from the MNIST dataset by extracting spatial features and performing multi-class classification.
```
## Model Architecture
```
Input Layer (28×28 grayscale image) to take handwritten digit image as input
Convolutional Layer (Conv2D) to extract important features like edges, curves, and shapes
MaxPooling Layer to reduce image size and keep important features (downsampling)
Second Convolutional Layer to learn deeper patterns like digit strokes and structures
Second MaxPooling Layer to further reduce dimensionality and improve efficiency
Flatten Layer to convert extracted feature maps into a 1D vector
Dense (Fully Connected) Layer to learn high-level digit representations
Dropout Layer to reduce overfitting by randomly dropping neurons during training
Softmax Output Layer for multi-class classification (predicting digits from 0 to 9)
```
## Key Features
```
Automatic Feature Extraction :CNN extracts edges, curves, shapes automatically (no manual feature engineering).
High Accuracy Classification : Classifies digits from 0 to 9 with strong performance.
Fast Training on MNIST Dataset : MNIST is lightweight, so training is efficient.
Dropout Regularization : Prevents overfitting and improves generalization.
Prediction on New Images : Model can predict unseen handwritten digits.
Visualization Support : Training & validation accuracy graphs are plotted.
```
## Tech Stacks
```
Python
TensorFlow
Keras
NumPy
Matplotlib
Convolutional Neural Network (CNN)
MNIST Handwritten Digits Dataset
```
## Results
```
Training(Epochs: 10 ,Optimizer: Adam ,Loss Function: Sparse Categorical Crossentropy)
Accuracy Achieved (Training Accuracy: ~99% ,Testing Accuracy: ~98% to 99% )
Loss : Test loss becomes very low (near 0.05 – 0.10)
Exact accuracy may slightly vary depending on training.
```
