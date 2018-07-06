import cupy as np
from datasets.planar import load_planar_dataset

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):
    # Retrieve parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Implement Forward Propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1/(1+np.exp(-Z2))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]

    # Compute the cross-entropy cost
    cost = - (1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    cost = np.squeeze(cost)

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    # Retrieve W2 from parameters
    W2 = parameters['W2']
        
    # Retrieve A1 and A2 from cache
    A1 = cache['A1']
    A2 = cache['A2']
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(params, grads, learning_rate = 1.5):
    return {'W1': params['W1'] - learning_rate * grads['dW1'],
            'b1': params['b1'] - learning_rate * grads['db1'],
            'W2': params['W2'] - learning_rate * grads['dW2'],
            'b2': params['b2'] - learning_rate * grads['db2']}

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
         
        A2, cache = forward_propagation(X, parameters)

        if print_cost and i % 1000 == 0:
            cost = compute_cost(A2, Y, parameters)
            print ("Cost after iteration %i: %f" %(i, cost))

        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

    return parameters

def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions

X, Y = load_planar_dataset()

# Convert dataset to cupy
X = np.array(X)
Y = np.array(Y)

parameters = nn_model(X, Y, n_h = 5, num_iterations = 5000, print_cost=True)
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')