import numpy as np
from LR import LinearRegression  # Assuming your main class is in a file called linear_regression.py
import unittest

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([1, 2, 3, 4, 5])
        self.model = LinearRegression()

    def test_fitting(self):
        # Test the fit function
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.coefficients, "The coefficients should be initialized after fitting the model")

    def test_predict(self):
        # Test the predict function
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        expected_predictions = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=5, 
                                             err_msg="Predictions do not match expected values")

    def test_r_squared(self):
        # Test the Rsquared function
        self.model.fit(self.X, self.y)
        r2 = self.model.Rsquared(self.X, self.y)
        self.assertAlmostEqual(r2, 1.0, places=5, msg="R-squared value should be close to 1 for perfect linear fit")

    def test_overall_performance(self):
        # Ensure the overall performance is satisfactory (low error, high R-squared)
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        r2 = self.model.Rsquared(self.X, self.y)

        # Check that predictions are close to actual values
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=5, 
                                             err_msg="Overall performance is not sufficient; predictions deviate too much")
        # Check that R-squared is high
        self.assertGreaterEqual(r2, 0.99, "R-squared should be close to 1 for good performance")

if __name__ == '__main__':
    unittest.main()
