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
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        
        # Initialize coefficients with zeros
        self.coefficients = np.zeros(X.shape[1])
        
        # Gradient Descent
        for i in range(self.n_iterations):
        #     Calculate predictions
            predictions = np.dot(X, self.coefficients)
        #     Calculate the error
            errors = predictions - y
        #     Calculate Mean Squared Error (MSE)
            mse  = np.mean(errors**2)
        #     Calculate the gradient of the error
            gradients = 2 * np.dot(X.T,errors) / len(X)
        #     Update coefficients using the learning rate
            self.coefficients -= self.learning_rate * gradients
        #     Log iteration and error
            self.log['iterations'].append(i)
            self.log['error'].append(mse)
        #     Check for early stopping
            if early_stop and i > 0:
                # Stop if the change in error is below the specified tolerance
                if abs(self.log['error'][-2] - mse) < self.tolerance:
                    self.log['early_stop'] = 'Yes'
                    break
        
    def train_SGD(self, X, y, early_stop=False, batch_size=1):
        # Add a column of ones for the intercept (bias term)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        # Initialize coefficients with zeros
        self.coefficients = np.zeros(X.shape[1])

        # Stochastic Gradient Descent
        for i in tqdm(range(self.n_iterations)):
            # Shuffle data for mini-batch SGD
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Iterate over batches
            for j in range(0, X.shape[0], batch_size):
                # Select a batch of data
                X_batch = X[j:j + batch_size]
                y_batch = y[j:j + batch_size]
                
                # Calculate predictions for the batch
                predictions = np.dot(X_batch, self.coefficients)
                # Calculate the error for the batch
                errors = predictions - y_batch
                # Calculate the gradient for the batch
                gradients = 2 * np.dot(X_batch.T, errors) / len(X_batch)

                # Update coefficients using the gradient and learning rate
                self.coefficients -= self.learning_rate * gradients

            # Calculate the mean squared error on the whole dataset
            predictions = np.dot(X, self.coefficients)
            mse = np.mean((predictions - y) ** 2)

            # Log the iteration number and error for tracking
            self.log['iterations'].append(i)
            self.log['error'].append(mse)

            # Check for early stopping if enabled
            if early_stop and i > 0:
                # Stop if the change in error is below the specified tolerance
                if abs(self.log['error'][-2] - mse) < self.tolerance:
                    self.log['early_stop'] = 'Yes'
                    break


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
        print("Successfully implemented the training method for Gradient Descent.")
    except NotImplementedError as e:
        print(e)
    
    '''
    try:
        model_1.train_SGD(X_train, y_train, early_stop=True, batch_size=32)
    except NotImplementedError as e:
        print(e)
    '''