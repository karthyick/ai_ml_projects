# My Python Projects

This repository contains various Python projects focused on machine learning and data analysis. Each project is located in its own directory or file, and below are the details for each project along with installation and usage instructions.

## Projects Overview

### 1. [DecisionTree.py](./DecisionTree.py)

- **Description:** This project implements a decision tree algorithm for a classification or regression task. It uses the scikit-learn library to create a decision tree model, train it, and evaluate its performance.
- **Technologies Used:** Python, scikit-learn, pandas, numpy
- **Features:**
    - Build a decision tree classifier or regressor
    - Visualize the tree structure
    - Evaluate model accuracy

### 2. [Diabetes_Prediction.ipynb](./Diabetes_Prediction.ipynb)

- **Description:** A Jupyter Notebook project for predicting diabetes using machine learning algorithms. This project uses historical health data to predict whether a patient will have diabetes.
- **Technologies Used:** Python, pandas, numpy, scikit-learn, Jupyter Notebook
- **Features:**
    - Data preprocessing and feature selection
    - Model training using classifiers like logistic regression, decision trees, etc.
    - Model evaluation with metrics like accuracy, precision, and recall

### 3. [Rock_vs_Mine_Prediction.ipynb](./Rock_vs_Mine_Prediction.ipynb)

- **Description:** A Jupyter Notebook project that classifies rock and mine data. It uses machine learning models to predict whether a given sample is a rock or a mine based on features extracted from data.
- **Technologies Used:** Python, pandas, numpy, scikit-learn, Jupyter Notebook
- **Features:**
    - Data analysis and exploration
    - Model selection and tuning
    - Predictions using classification algorithms

### 4. [diabetes_prediction.py](./diabetes_prediction.py)

- **Description:** A Python script that automates the diabetes prediction model. It loads a dataset, preprocesses it, and uses machine learning models to make predictions about diabetes outcomes.
- **Technologies Used:** Python, pandas, numpy, scikit-learn
- **Features:**
    - Command-line script to run predictions
    - Model training and testing
    - Results saved to a file

### 5. [rock_vs_mine_prediction.py](./rock_vs_mine_prediction.py)

- **Description:** A Python script version of the rock vs mine prediction project that automates the classification process using machine learning models. It takes input data, processes it, and predicts whether the sample is a rock or a mine.
- **Technologies Used:** Python, pandas, numpy, scikit-learn
- **Features:**
    - Loads data and performs model predictions
    - Command-line interface for easy use
    - Model evaluation and output display
    - 
### 6. [GridSearchCV_and_RandomizedSearchCV.ipynb](./GridSearchCV_and_RandomizedSearchCV.ipynb)

- **Description:** A Jupyter Notebook that demonstrates hyperparameter tuning using Grid Search and Randomized Search for a Decision Tree Classifier on different datasets. The project includes model evaluation and visualization of performance.
- **Technologies Used:** Python, pandas, numpy, scikit-learn, matplotlib, Jupyter Notebook
- **Features:**
    - Loads Hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
    - Classification performance evaluation using accuracy, classification report, and confusion matrix.
    - Visualizations - evaluationConfusion Matrix heatmap to analyze correct classifications (diagonal values) and misclassifications (off-diagonal values).
    - Supports multiple datasets (e.g., Iris dataset).


## Getting Started

### Requirements

Each project has its own dependencies. You can install them using the `requirements.txt` file for each project if provided.

1. Clone the repository:
    ```bash
    git clone https://github.com/username/repository-name.git
    ```

2. Navigate to the project directory:
    ```bash
    cd repository-name
    ```

3. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. Install the project dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Each project can be run individually by navigating to its respective directory and running the script or notebook:

- **For DecisionTree.py:**
    ```bash
    python DecisionTree.py
    ```

- **For Diabetes_Prediction.ipynb:**
    Open the notebook in Jupyter and run the cells:
    ```bash
    jupyter notebook diabetes_prediction/Diabetes_Prediction.ipynb
    ```

- **For Rock_vs_Mine_Prediction.ipynb:**
    Open the notebook in Jupyter and run the cells:
    ```bash
    jupyter notebook rock_vs_mine_prediction/Rock_vs_Mine_Prediction.ipynb
    ```

- **For diabetes_prediction.py:**
    ```bash
    python diabetes_prediction/diabetes_prediction.py
    ```

- **For rock_vs_mine_prediction.py:**
    ```bash
    python rock_vs_mine_prediction/rock_vs_mine_prediction.py
    ```
