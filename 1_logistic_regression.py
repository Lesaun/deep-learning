import cupy as cp
import numpy as np
import h5py
import scipy

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def propagate(w, b, X, Y):
    m = X.shape[1]

    ## Forward propagate
    Z = cp.dot(w.T, X) + b
    A = 1 / (1 + cp.exp(-Z))
    cost = - (1 / m) * cp.sum(Y * cp.log(A) + (1 - Y) * cp.log(1 - A))
    cost = cp.squeeze(cost)

    ## Backward propagate
    dZ = A - Y
    dw = (1 / m) * cp.dot(X, dZ.T)
    db = (1 / m) * cp.sum(dZ)

    grads = {'dw': dw, 'db': db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        w = w - learning_rate * grads['dw']
        b = b - learning_rate * grads['db']

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': grads['dw'], 'db': grads['db']}

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]

    Y_prediction = cp.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    Z = cp.dot(w.T, X) + b
    A = 1 / (1 + cp.exp(-Z))

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 0 if A[0, i] <= 0.5 else 1

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 3000, learning_rate = 0.5, print_cost = False):
    ## Initialize parameters
    w = cp.zeros((X_train.shape[0], 1))
    b = 0

    # Gradient descent (≈ 1 line of code)
    parameters, _, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - cp.mean(cp.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - cp.mean(cp.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

## Load data set
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

## Convert to cupy (bug preventing direct from h5 to cp)
train_set_x_orig = cp.array(train_set_x_orig)
train_set_y = cp.array(train_set_y)
test_set_x_orig = cp.array(test_set_x_orig)
test_set_y = cp.array(test_set_y)

## Flatten training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

## Standardize dataset (change color in pixels to percentages)
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

d = model(train_set_x,
          train_set_y,
          test_set_x,
          test_set_y,
          num_iterations = 10000,
          learning_rate = 0.005,
          print_cost = True)