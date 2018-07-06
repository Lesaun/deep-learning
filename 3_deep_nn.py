import cupy as np
from datasets.catvnotcat import load_dataset

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def layer_forward(A_prev, W, b, activation):
    Z = W.dot(A_prev) +  b

    if activation == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))
    
    elif activation == 'relu':
        A = np.maximum(0, Z)

    cache = (A_prev, W, b, Z)

    return A, cache

def model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    # forward prop layers 1..L-1
    for l in range(1, L):
        A_prev = A 
        A, cache = layer_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = 'relu')
        caches.append(cache)
    
    # forward prop layer L
    AL, cache = layer_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    return cost

def layer_backward(dA, cache, activation):
    (A_prev, W, _, Z) = cache
    m = A_prev.shape[1]
    
    if activation == 'relu':
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        
    elif activation == 'sigmoid':
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
    
    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Backprop first layer
    L_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = layer_backward(dAL, L_cache, activation = 'sigmoid')
    
    # Backprop layers L-1..1
    for l in reversed(range(1, L)):
        l_cache = caches[l - 1]
        dA_prev_temp, dW_temp, db_temp = layer_backward(grads["dA" + str(l)], l_cache, activation = 'relu')
        grads["dA" + str(l - 1)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters

def model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    parameters = initialize_parameters(layers_dims)
    
    for i in range(0, num_iterations):
        AL, caches = model_forward(X, parameters)

        cost = compute_cost(AL, Y)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

        grads = model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
                
    return parameters

def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1,m))

    # Forward propagation
    probas, _ = model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((p == y)/m)))

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# Convert to cupy
train_x_orig = np.array(train_x_orig)
train_y = np.array(train_y)
test_x_orig = np.array(test_x_orig)
test_y = np.array(test_y)

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

layers_dims = [12288, 40, 20, 7, 5, 1]
parameters = model(train_x, train_y, layers_dims, num_iterations = 8000, print_cost = True)

print()
print("Train set: ")
predict(train_x, train_y, parameters)
print()
print("Test set: ")
predict(test_x, test_y, parameters)
