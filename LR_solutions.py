import numpy as np

class LinearRegression:
    def __init__(self):
        # Initialize the coefficients and intercept, which will be calculated after fitting the model
        self.intercept = None
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones for the intercept (bias term)
        ones = np.ones((len(X), 1))
        X = np.concatenate((ones, X), axis=1)

        # Calculate the Normal Equation
        XT = X.T  # Transpose of X
        XTX = XT.dot(X)  # X^T * X
        XTX_inv = np.linalg.inv(XTX)  # Inverse of (X^T * X)
        XTy = XT.dot(y)  # X^T * y
        self.coefficients = XTX_inv.dot(XTy)  # Calculate the coefficients

        #raise NotImplementedError("The fit method is not yet implemented")

    def predict(self, X):
        # Add a column of ones to match the structure used in fitting the model
        ones = np.ones((len(X), 1))
        X = np.concatenate((ones, X), axis=1)
        
        # Use the calculated coefficients to make predictions
        return X.dot(self.coefficients)

    def Rsquared(self, X, y):
        # Calculate R-squared to evaluate model performance
        ypred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
        ss_residual = np.sum((y - ypred)**2)  # Residual sum of squares
        return 1 - (ss_residual / ss_total)  # R-squared formula


if __name__ == "__main__":
    # Sample data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])

    # Initialize and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions and calculate R-squared
    ypred = model.predict(X)
    print(ypred)  # Predicted values

    print(model.Rsquared(X, y))  # R-squared value
