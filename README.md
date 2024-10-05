# Implementing the LinearRegression Class

## Getting Started
Before diving into the code, you need to set up your environment:

1. Clone the repository where this project is hosted:
   ```bash
   git clone https://github.com/HarrySSH/LinearModels.git
   cd LinearModels
   ```
2. Create a Conda environment by running:
   ```bash
   conda env create -f regression_environment.yml
   ```
3. Activate the new environment:
   ```bash
   conda activate regression_env
   ```



## Overview
Welcome to your task of completing the implementation of a `LinearRegression` class using Python and NumPy. The goal of this exercise is to deepen your understanding of the linear regression algorithm by manually coding the methods used to fit the model to the data, make predictions, and evaluate the model's performance.

## Method Descriptions

### 1. `__init__`
**Purpose**: Initializes the LinearRegression instance.
**Tasks**:
  - Initialize `self.coefficients` to `None`. This will later hold the coefficients (weights) calculated from the fit method.
  - Initialize `self.intercept` to `None`, which will store the intercept from the regression model.

### 2. `fit`
**Purpose**: Fits the linear regression model to the provided data.
**Tasks**:
  - Add a column of ones to the input feature matrix `X` to account for the intercept in the linear model.
  - Compute the transpose of matrix `X`.
  - Calculate the product of the transpose of `X` and `X` itself.
  - Compute the inverse of this product.
  - Calculate the product of the transpose of `X` and the target vector `y`.
  - Solve for the coefficient vector using the Normal Equation (`XTX_inv * XTy`). This vector includes the intercept as its first element.

### 3. `predict`
**Purpose**: Makes predictions using the linear model.
**Tasks**:
  - Add a column of ones to the input feature matrix `X` if it was not included during fitting.
  - Compute the dot product of the feature matrix `X` and the coefficients (including intercept) to predict the target variable.

### 4. `Rsquared`
**Purpose**: Calculates the R-squared value to evaluate the model performance.
**Tasks**:
  - Use the `predict` method to obtain predictions for the input feature matrix `X`.
  - Calculate the total sum of squares (variation of `y` from its mean).
  - Calculate the residual sum of squares (variation of `y` from the predicted values).
  - Compute the R-squared value using the formula `1 - (residual sum of squares / total sum of squares)`.

## Completion Guide
To complete this exercise, follow these steps:
1. Read and understand the purpose of each method and what it is supposed to accomplish.
2. Start by implementing the `fit` method as it will compute the necessary coefficients used by other methods.
3. Implement the `predict` method to make use of the coefficients computed in `fit`.
4. Implement the `Rsquared` method to evaluate the model's performance using the predictions.
5. Test each method as you implement them to ensure correctness.

## Testing Your Implementation
Once you have implemented the methods, you can test your class using pytest. Run the following command in your terminal:
```bash
pytest
```
This command will execute the test cases defined in the pytest test suite. Ensure that your functions are correctly implemented by passing all the tests.

Further Reading for unit testing.
[text](https://docs.pytest.org/en/stable/)
