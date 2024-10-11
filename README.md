# Linear Models Homework Assignment

This project involves implementing and training linear regression models using gradient descent and stochastic gradient descent (SGD). You will start by completing the provided Python script and then run corresponding sections of the Jupyter notebook to validate your work.

## Getting Started
Before diving into the code, you need to set up your environment:

1. Clone the repository where this project is hosted:
   ```bash
   git clone github_repo
   cd LinearModels_GD
   ```
2. Create a Conda environment by running:
   ```bash
   conda env create -f lm_env.yaml
   ```
3. Activate the new environment:
   ```bash
   conda activate lm_env
   ```

## Workshop Instructions

### Step 1: Implement the `train` Method

- Start by completing the `train` method in the Python script (`LinearModels.py`). The `train` method uses standard gradient descent to minimize the error between predicted and actual values.
- Refer to the comments and pseudocode provided in the script to guide your implementation.

After completing the `train` method:

1. Open the Jupyter notebook `LM_playground.ipynb`.
2. Run sections 1.1 to 1.3 to validate your implementation.
   - These sections involve training the linear model using your implementation and play with the learning rate. 

### Step 2: Implement the `train_SGD` Method

- Once you have successfully implemented and tested the `train` method, proceed to implement the `train_SGD` method. This method uses Stochastic Gradient Descent (SGD) to train the linear model.
- As before, the script includes comments and pseudocode to guide your work.

After completing the `train_SGD` method:

1. Open the Jupyter notebook `LM_playground.ipynb`.
2. Run sections 2.1 to 2.3 to validate your implementation.
   - These sections will train the linear model using your SGD implementation and try to answer the question in the jupyter notebook.

## Notes
- Make sure to log the error and iteration details during training for both methods. This will help you observe the convergence behavior of the algorithms.
- The Jupyter notebook is structured to guide you through both implementations, including visualizations of the model's learning process.



