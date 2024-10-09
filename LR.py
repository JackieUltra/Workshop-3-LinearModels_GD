import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class LinearModels:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-6, momentum=0.0):
        # Initialize the coefficients and intercept
        self.coefficients = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.momentum = momentum
        # Initialize logging dictionary
        self.log = {'learning_rate': self.learning_rate, 'iterations': [], 'error': [], 'early_stop': 'No'}

    def fit(self, X, y):
        # Add a column of ones for the intercept (bias term)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        # Transpose of X
        self.X = X.T

        # X^T * X
        self.XTX = np.dot(self.X, X)

        # Inverse of (X^T * X)
        self.XTX_inv = np.linalg.pinv(self.XTX)
        
        # X^T * y
        self.XTy = np.dot(self.X, y)
        
        # Calculate the coefficients using the Normal Equation
        self.coefficients = np.dot(self.XTX_inv, self.XTy)

    def train(self, X, y, early_stop=False):
        # If momentum is set, call train_with_momentum
        if self.momentum > 0.0:
            return self.train_with_momentum(X, y, early_stop)
        
        # Add a column of ones for the intercept (bias term)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        
        # Initialize coefficients
        self.coefficients = np.zeros(X.shape[1])
        
        # Gradient Descent
        for i in range(self.n_iterations):
            predictions = np.dot(X, self.coefficients)
            errors = predictions - y
            mse = np.mean(errors**2)  # Mean squared error
            gradients = 2 * np.dot(X.T, errors) / len(X)
            
            # Update coefficients
            self.coefficients -= self.learning_rate * gradients
            
            # Log the iteration and error
            self.log['iterations'].append(i)
            self.log['error'].append(mse)
            
            # Check for early stopping
            if early_stop and i > 0:
                if abs(self.log['error'][-2] - mse) < self.tolerance:
                    self.log['early_stop'] = 'Yes'
                    break

    def train_with_momentum(self, X, y, early_stop=False):
        # Add a column of ones for the intercept (bias term)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        
        # Initialize coefficients and velocity
        self.coefficients = np.zeros(X.shape[1])
        velocity = np.zeros(X.shape[1])
        
        # Gradient Descent with Momentum
        for i in range(self.n_iterations):
            predictions = np.dot(X, self.coefficients)
            errors = predictions - y
            mse = np.mean(errors**2)  # Mean squared error
            gradients = 2 * np.dot(X.T, errors) / len(X)
            
            # Update velocity and coefficients
            velocity = self.momentum * velocity - self.learning_rate * gradients
            self.coefficients += velocity
            
            # Log the iteration and error
            self.log['iterations'].append(i)
            self.log['error'].append(mse)
            
            # Check for early stopping
            if early_stop and i > 0:
                if abs(self.log['error'][-2] - mse) < self.tolerance:
                    self.log['early_stop'] = 'Yes'
                    break

    def predict(self, X):
        # Add a column of ones to match the structure used in fitting the model
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        # Use the calculated coefficients to make predictions
        y_pred = np.dot(X, self.coefficients)
        return y_pred

    def Rsquared(self, X, y):
        # Predict y values using the predict method
        # Total sum of squares
        total_sum_squares = np.sum((y - np.mean(y)) ** 2)

        # Residual sum of squares
        residuals = y - self.predict(X)

        # R-squared formula
        r_squared = 1 - (np.sum(residuals ** 2) / total_sum_squares)
        return r_squared

if __name__ == "__main__":
    # Sample data
    X_multi, y_multi = X_multi, y_multi = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

    # Initialize and train the model using gradient descent
    """
    model = LinearModels(learning_rate=1, n_iterations=1000)
    # random split it into train and Test
    X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
    
    model.train(X_train, y_train)
   
    # Make predictions and calculate R-squared
    # Predicted values
    print("R squared score on the X_test set ")
    print(model.Rsquared(X_test, y_test))

    # Plot error over iterations
    import matplotlib.pyplot as plt
    plt.plot(model.log['iterations'], model.log['error'], label='Mean Squared Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Progression')
    plt.legend()
    plt.show()
    plt.close()
    """
    print("?")
    import matplotlib.pyplot as plt
    model = LinearModels(learning_rate=1, n_iterations=1000)
    # random split it into train and Test
    X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)


    # with iteration as 1000
    # plot the error rate from 1, 0.1, 0.01, 0.001, 0.0001 in different color
    # and show the plot
    model = LinearModels(learning_rate=1, n_iterations=1000)
    model.train(X_train, y_train)
    ylim = model.log['error'][0]
    plt.plot(model.log['iterations'], model.log['error'], label='LR - 1', color='red')
    model = LinearModels(learning_rate=0.1, n_iterations=1000)
    model.train(X_train, y_train)
    
    ylim = max(ylim, model.log['error'][0])
    
    plt.plot(model.log['iterations'], model.log['error'], label='LR - 0.1', color='blue')
    model = LinearModels(learning_rate=0.01, n_iterations=1000)
    model.train(X_train, y_train)
    ylim = max(ylim, model.log['error'][0])
    
    plt.plot(model.log['iterations'], model.log['error'], label='LR - 0.01', color='green')
    model = LinearModels(learning_rate=0.001, n_iterations=1000)
    model.train(X_train, y_train)
    ylim = max(ylim, model.log['error'][0])
    plt.plot(model.log['iterations'], model.log['error'], label='LR - 0.001', color='yellow')
    model = LinearModels(learning_rate=0.0001, n_iterations=1000)
    model.train(X_train, y_train)
    ylim = max(ylim, model.log['error'][0])
    plt.plot(model.log['iterations'], model.log['error'], label='LR - 0.0001', color='black')

    print("Why")
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Progression')
    plt.ylim(0, ylim+5000)
    plt.legend()
    plt.show()

