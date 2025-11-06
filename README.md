Neural Network Digit Recognizer (NumPy Only)
📌 Overview

This project implements a fully connected neural network from scratch, using only NumPy, to classify handwritten digits from the MNIST Digit Recognizer Dataset.
The goal is to demonstrate how forward propagation, backpropagation, gradient descent, activation functions, and loss computation work internally—without relying on deep-learning libraries like TensorFlow or PyTorch.

✨ Features

✅ Built completely from scratch using NumPy
✅ Works on the Kaggle Digit Recognizer dataset
✅ Supports 2 hidden layers
✅ Uses ReLU activation for hidden layers
✅ Uses Softmax for output
✅ Uses Cross-Entropy Loss
✅ Trains using batch gradient descent
✅ Outputs loss every 100 epochs
✅ Easy-to-understand code structure

🧠 Neural Network Architecture

The network uses the following structure:

Layer	Type	Size
Input Layer	Flattened pixels	784 neurons
Hidden Layer 1	Dense + ReLU	64 neurons
Hidden Layer 2	Dense + ReLU	32 neurons
Output Layer	Dense + Softmax	10 neurons (digits 0-9)
📂 Dataset

You must download the Kaggle MNIST dataset:

train.csv — contains 42k images + labels

test.csv — contains images (for final evaluation)

Each image is 28×28 pixels flattened into 784 columns.

🚀 Training Workflow

Load dataset using Pandas

Normalize pixel values to [0,1]

Convert labels into one-hot encoded vectors

Initialize weights & biases

Perform:

Forward propagation

Loss calculation

Backpropagation

Gradient descent update

Repeat for the given number of epochs

🛠️ Technologies Used

NumPy → Total math backend

Pandas → Data loading

Matplotlib (optional) → Visualization

No ML libraries are used—everything is handcrafted.

📈 Example Training Output
Epoch 0   | Loss: 2.3015
Epoch 100 | Loss: 0.7421
Epoch 200 | Loss: 0.4238
Epoch 300 | Loss: 0.2890
...

📑 How to Run
pip install numpy pandas matplotlib
python train_digit_recognizer.py


Make sure your dataset path (train.csv) is correct.

📌 Future Improvements

Add mini-batch training

Add Adam optimizer

Add Dropout

Evaluate accuracy on test set

Save & load model weights
