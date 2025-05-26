# Neural-network-
this is a digit recognizer artificial intelligence training module using the mnist database along with forward and backword propagation and updating the weights as required 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
from matplotlib import pyplot as pt

data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
print(data)

row,col=data.shape
print(row,col)

rng=np.random.default_rng()

def weights(rows,cols):
    weights=rng.random((rows,cols))
    print(weights)

weights(row,col)

weights_input_hidden=np.random.randn(784,64)
print("----")
print("----")
weights_hidden_hidden2=np.random.randn(64,32)
print("----")
print("----")
weights_hidden_output=np.random.randn(32,10)
print("----")
print("----")

bias_hidden1 = np.zeros(64)  # Bias for first hidden layer (64 neurons)
bias_hidden2 = np.zeros(32)  # Bias for second hidden layer (32 neurons)
bias_output = np.zeros(10)  # Bias for output layer (10 neurons)

learning_rate=0.04

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):  # You were missing this!
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(x,y):
    eps=1e-8
    loss=-np.mean(np.sum(x*np.log(y+eps),axis=0))
    return loss

train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
X_train = train_df.drop('label', axis=1).values / 255.0  # Normalize input
y_train = train_df['label'].values

y_onehot = np.zeros((y_train.size, 10))
y_onehot[np.arange(y_train.size), y_train] = 1

actual_output=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

epoche=1000

for i in range(epoche):
    hidden1=relu(np.dot(X_train,weights_input_hidden)+bias_hidden1)
    hidden2=relu(np.dot(hidden1,weights_hidden_hidden2)+bias_hidden2)
    predicted_output=softmax(np.dot(hidden2,weights_hidden_output)+bias_output)

x_train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
x_data=x_train.drop('label').values/255.0
y_data=x_train['label'].values

y_oneshot=np.zeros((y_train.size,10))
y_train[np.arrange(y_train.size,y_train)]=1

loss=cross_entropy_loss(y_oneshot,predicted_output)

for i in range(1000):
    error_loss=predicted_output-y_oneshot
    grad_weigths_output=np.dot(hidden2,error_loss)
    grad_biases_output=np.sum(error_loss)
    
    error_hidden2=np.dot(error_loss,weight_hidden_output.T)
    relu_derivative(error_hidden2)
    grad_weight_2=np.dot(hidden1.T,error_hidden2)
    grad_biases_2=np.sum(error_hidden2,axis=1)
    
    error_hidden1=np.dot(error_loss,weight_hidden_hidden2.T)
    relu_derivative(error_hidden1)
    grad_weight_1=np.dot(x_train,error_hidden1)
    grad_biases_1=np.sum(error_hidden1,axis=1)

    weight_update_input-=(learning_rate*error_hidden1)
    weight_update_hidden-=(learning_rate*error_hidden2)
    weight_update_output-=(learning_rate*error_ouput)
    
    
    biases_update_input-=(learning_rate*grad_biases_1)
    biases_update_hidden-=(learning_rate*grad_biases_2)
    biases_update_output-=(learning_rate*grad_biases_output)
    

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

