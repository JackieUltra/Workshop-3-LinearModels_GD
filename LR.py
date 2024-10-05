import numpy as np

class LinearRegression:
    def __init__(self):
        # Initialize the coefficients and intercept with None
        #  which will be calculated after fitting the model
        #self.intercept = 
        #self.coefficients = 

        pass # after you have added the above lines, you can remove this line

    def fit(self, X, y):
        # Add a column of ones for the intercept (bias term)

        # Transpose of X

        # X^T * X

        # Inverse of (X^T * X)
        
        # X^T * y
        
        # Calculate the coefficients using the Normal Equation
        raise NotImplementedError("The fit method is not yet implemented")  # after you have added the above lines, you can remove this line

    def predict(self, X):
        # Add a column of ones to match the structure used in fitting the model

        # Use the calculated coefficients to make predictions

        raise NotImplementedError("The predict method is not yet implemented") # after you have added the above lines, you can remove this line

    def Rsquared(self, X, y):
        # Predict y values using the predict method
        # Total sum of squares

        # Residual sum of squares

        # R-squared formula

        raise NotImplementedError("The Rsquared method is not yet implemented") # after you have added the above lines, you can remove this line

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
