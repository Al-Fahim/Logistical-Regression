import numpy as np

#Load the data from directory
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

#Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Score Functions
def score(X, w, b):
    z = np.dot(X, w) + b #dot product of wX + b
    return z

#Cross Entropy function
def cross_entropy_loss(Y, y_pred):
    return -np.mean(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))

#Prediction Function
def predict(X, w, b):
    scores = score(X, w, b)
    return sigmoid(scores)

#Gradient Function
def compute_gradient(X, y, y_hat):
    m = X.shape[0]
    dW = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    return dW, db

#Train the model
def logistic_regression(X_train, Y_train, learning_rate=0.01, num_iterations=1000):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0

    for i in range(num_iterations):
        #Forward pass: make predictions and calculate loss
        y_pred = predict(X_train, w, b)
        loss = cross_entropy_loss(Y_train, y_pred)

        #Backward pass: calculate gradients and update weights and bias
        dw, db = compute_gradient(X_train, Y_train, y_pred)
        w -= learning_rate * dw
        b -= learning_rate * db

        #Print loss every 100 iterations
        if i % 100 == 0:
            print("Loss after iteration %i: %f" % (i, loss))
    return w, b

#Test the model
def test_logistic_regression():
    #Train on the data
    w, b = logistic_regression(X_train, Y_train)

    #Make predictions on the test data
    y_pred = predict(X_test, w, b)
    y_pred = np.round(y_pred)

    #Calculate accuracy
    accuracy = np.mean(Y_test == y_pred)
    print("Accuracy: %.2f" % accuracy)

#Run the program
test_logistic_regression()