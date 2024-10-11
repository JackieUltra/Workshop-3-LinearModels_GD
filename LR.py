import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class LinearModels:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-3):
        # Initialize the coefficients and intercept
        self.coefficients = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        # Initialize logging dictionary
        self.log = {'learning_rate': self.learning_rate, 'iterations': [], 'error': [], 'early_stop': 'No'}

    def fit(self, X, y):
        raise NotImplementedError("We do not need to implement the fit method for this assignment.")
        # Add a column of ones for the intercept (bias term)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        # Compute X^T * X
        XTX = np.dot(X.T, X)

        # Compute the inverse of (X^T * X)
        XTX_inv = np.linalg.pinv(XTX)
        
        # Compute X^T * y
        XTy = np.dot(X.T, y)
        
        # Calculate the coefficients using the Normal Equation: (X^T * X)^(-1) * X^T * y
        self.coefficients = np.dot(XTX_inv, XTy)

    def train(self, X, y, early_stop=False):
        """
        Train the linear model using Gradient Descent.
        This method should be implemented by students as part of the homework assignment.
        """
        # Add a column of ones for the intercept (bias term)
        # Pseudo Code: X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        
        # Initialize coefficients with zeros
        # Code: self.coefficients = np.zeros(X.shape[1])
        
        # Iterate through the number of iterations (self.n_iterations)
        # Pseudo Code: for i in range(self.n_iterations):
        #     Calculate predictions
        #     Calculate the error
        #     Calculate Mean Squared Error (MSE)
        #     Calculate the gradient of the error
        #     Update coefficients using the learning rate
        #     Log iteration and error
        #     Check for early stopping
        
        ######### You should not have to look at the code below this line#########
        ######### This is a line by line explanation of the code above #########
        ######### For line where I say UNCOMMENT THIS LINE, you should write code there and uncomment that line #########
        # Gradient Descent
        # UNCOMMENT THIS LINE: You need a for loop to iterate through the number of iterations

            # Calculate predictions: X * coefficients
            # UNCOMMENT THIS LINE: You should use np.dot to calculate the predictions

            # Calculate the error between predictions and actual values
            # UNCOMMENT THIS LINE: You should subtract the actual values from the predictions

            # Calculate Mean Squared Error (MSE)
            # UNCOMMENT THIS LINE: You should calculate the mean of the squared errors, just do the mean of the square of the errors

            # Calculate the gradient of the error with respect to coefficients
            # UNCOMMENT THIS LINE: You should calculate the gradient using the formula: grads= 2 * np.dot(X.T, errors) / len(X)
            
            # Update coefficients using the gradient and learning rate
            # UNCOMMENT THIS LINE: You should update the coefficients by subtracting the product of the learning rate and the gradient from the coefficients
            
            # Log the iteration number and error for tracking
            # UNCOMMENT THIS LINE: You should log the iteration number and error in the log dictionary with append method
            
            # Check for early stopping if enabled
            #if early_stop and i > 0:
            #    # Stop if the change in error is below the specified tolerance
            #    if abs(self.log['error'][-2] - mse) < self.tolerance:
            #        self.log['early_stop'] = 'Yes'
            #        break

        raise NotImplementedError("Gradient Descent training method needs to be implemented by students.")

    def train_SGD(self, X, y, early_stop=False, batch_size=1):
        """
        Train the linear model using Stochastic Gradient Descent.
        This method should be implemented by students as part of the homework assignment.
        """
        # Add a column of ones for the intercept (bias term)
        # Pseudo Code: X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        
        # Initialize coefficients with zeros
        # Code: self.coefficients = np.zeros(X.shape[1])
        
        # Iterate through the number of iterations (self.n_iterations)
        # Pseudo Code: for i in range(self.n_iterations):
        #     Shuffle the dataset
        #     Iterate over batches
        #     For each batch, calculate predictions and error
        #     Calculate the gradient for the batch
        #     Update coefficients using the gradient and learning rate
        #     Log iteration and error
        #     Check for early stopping if enabled
        

        raise NotImplementedError("Stochastic Gradient Descent training method needs to be implemented by students.")

    def predict(self, X):
        # Add a column of ones to match the structure used in fitting the model
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        # Use the calculated coefficients to make predictions: X * coefficients
        y_pred = np.dot(X, self.coefficients)
        return y_pred

    def Rsquared(self, X, y):
        raise NotImplementedError("We do not need to implement the Rsquared method for this assignment.")
        # Predict y values using the predict method
        y_pred = self.predict(X)
        # Calculate the total sum of squares (TSS)
        total_sum_squares = np.sum((y - np.mean(y)) ** 2)
        # Calculate the residual sum of squares (RSS)
        residuals = y - y_pred
        # Calculate R-squared value
        r_squared = 1 - (np.sum(residuals ** 2) / total_sum_squares)
        return r_squared

if __name__ == "__main__":
    # Generate a synthetic dataset for regression
    X, y = make_regression(n_samples=10000, n_features=100, noise=30, random_state=42)

    # Reshape y to be a 1D array
    y = y.reshape(-1)

    # Split the dataset into training and testing sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    model_1 = LinearModels(learning_rate=0.01, n_iterations=1000)
    
    # Training is not implemented, as it is assigned for students to complete.
    try:
        model_1.train(X_train, y_train, early_stop=True)
    except NotImplementedError as e:
        print(e)
    print("Successfully implemented the training method for Gradient Descent.")
    '''
    try:
        model_1.train_SGD(X_train, y_train, early_stop=True, batch_size=32)
    except NotImplementedError as e:
        print(e)
    '''