# Handwritten-Number-Prediction
This project is focused on building a machine learning model that can recognize and predict handwritten numbers. The model is built using Python and TensorFlow.

## Dataset
The dataset used for this project is the MNIST dataset which contains 60,000 training images and 10,000 testing images of handwritten digits.

## Model Architecture
The model architecture used in this project is a convolutional neural network (CNN) which is a type of deep learning model that is commonly used for image recognition tasks. The CNN consists of multiple convolutional layers and pooling layers, followed by a few fully connected layers at the end. The architecture used for this project is as follows:
Input -> Conv2D -> MaxPool2D -> Conv2D -> MaxPool2D -> Flatten -> Dense -> Dropout -> Output

## Training the Model
The model is trained using the MNIST dataset. The training process involves passing the training images through the model and adjusting the weights to minimize the difference between the predicted and actual values. The training process can take several hours depending on the complexity of the model and the size of the dataset.

## Testing the Model
Once the model is trained, it is tested using the testing images from the MNIST dataset. The testing process involves passing the testing images through the model and calculating the accuracy of the model based on the predicted and actual values.
