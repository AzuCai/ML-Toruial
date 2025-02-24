# Machine Learning Algorithms Tutorial

## Project Description
This repository contains a collection of Python-based implementations of classic machine learning algorithms, created by me during my time as a Teaching Assistant (TA) for a Machine Learning course. The purpose of this tutorial is to provide hands-on learning material for students and enthusiasts to understand the inner workings of these algorithms through clear, handwritten code. Each algorithm is implemented from scratch (without relying on pre-built libraries like scikit-learn for the core logic), applied to real-world datasets, and includes visualization or evaluation of results.

This project is designed for educational use, offering a practical way to explore machine learning concepts, from supervised and unsupervised learning to ensemble methods.

## Prerequisites
To run the code in this tutorial, you need to install the following Python libraries:

NumPy: For numerical computations (pip install numpy)

Matplotlib: For visualization (pip install matplotlib)

Scikit-learn: For loading datasets and splitting data (pip install scikit-learn)

Seaborn: For loading the Titanic dataset in decision_tree.py (pip install seaborn)

Install all dependencies with:
```bash
pip install numpy matplotlib scikit-learn seaborn
```
## File Overview

### linear_regression.py
Dataset: Boston Housing (from sklearn.datasets.load_boston)

Task: Predict house prices based on features like number of rooms and crime rate.

Description: Implements linear regression using the normal equation method to fit a linear model to the data. Visualizes predicted vs. true house prices and computes Mean Squared Error (MSE).

### logistic_regression.py
Dataset: Breast Cancer Wisconsin (from sklearn.datasets.load_breast_cancer)

Task: Classify tumors as benign (0) or malignant (1) using tumor features like radius and texture.

Description: Implements logistic regression with gradient descent, including a numerically stable sigmoid function. Normalizes data to prevent overflow and visualizes predictions for the first two features.

### knn.py
Dataset: Digits (from sklearn.datasets.load_digits)

Task: Classify handwritten digits (0-9) based on pixel values.

Description: Implements K-Nearest Neighbors (KNN) with Euclidean distance and majority voting. Displays accuracy and visualizes predictions for a few test samples as images.

### naive_bayes.py
Dataset: Simulated spam dataset (using sklearn.datasets.make_classification)

Task: Classify emails as spam (1) or not spam (0) based on synthetic features.

Description: Implements Gaussian Naive Bayes, assuming features follow a Gaussian distribution. Computes class priors and likelihoods, then visualizes predictions for two synthetic features.

### decision_tree.py
Dataset: Titanic (from seaborn.load_dataset('titanic'))

Task: Predict whether a passenger survived (1) or not (0) based on age and fare.

Description: Implements a decision tree with Gini impurity for splitting. Builds a tree up to a specified depth and visualizes predictions for age and fare features.

### svm.py
Dataset: Iris (from sklearn.datasets.load_iris, binary subset)

Task: Classify iris flowers into two species (Setosa vs. Versicolor) using sepal length and width.

Description: Implements a simplified linear Support Vector Machine (SVM) with gradient descent. Visualizes the decision boundary and predictions for the first two features.

### perceptron.py
Dataset: Breast Cancer Wisconsin (from sklearn.datasets.load_breast_cancer)

Task: Classify tumors as benign (0) or malignant (1) using the first two features.

Description: Implements the Perceptron algorithm, a simple linear classifier. Normalizes data and visualizes predictions for mean radius and texture.

### kmeans.py
Dataset: Wine (from sklearn.datasets.load_wine)

Task: Cluster wines into 3 groups based on chemical properties (first two features).

Description: Implements K-Means clustering with random centroid initialization. Visualizes clusters and centroids for alcohol and malic acid features.

### pca.py
Dataset: Olivetti Faces (from sklearn.datasets.fetch_olivetti_faces)

Task: Reduce the dimensionality of face images to 2D for visualization.

Description: Implements Principal Component Analysis (PCA) using eigenvalue decomposition. Projects face data onto the top 2 principal components and visualizes the result.

### random_forest.py
Dataset: Wine (from sklearn.datasets.load_wine)

Task: Classify wines into 3 quality classes based on chemical features.

Description: Implements a Random Forest with multiple decision trees and feature subsampling. Normalizes data and visualizes predictions for the first two features.

### adaboost.py
Dataset: Breast Cancer Wisconsin (from sklearn.datasets.load_breast_cancer)

Task: Classify tumors as benign (0) or malignant (1) using the first two features.

Description: Implements AdaBoost with decision stumps as weak learners. Normalizes data and visualizes predictions for mean radius and texture.

## Usage
Ensure all required libraries are installed.

Run any file individually with: python filename.py

Each script will output a performance metric (e.g., accuracy, MSE) and display a visualization of the results.


## Educational Focus
The implementations prioritize clarity over optimization, making them ideal for learning but not production use.

## Datasets
Most datasets are from sklearn, with Titanic from seaborn. Simulated data is used where real datasets weren’t directly available in sklearn.

## Extensions
Feel free to replace simulated datasets (e.g., in naive_bayes.py) with real UCI datasets using pandas for more authenticity.

## Acknowledgments
This tutorial was developed as part of my responsibilities as a TA for a Machine Learning course during my PhD studies. Thanks to the course instructors and students for their feedback and inspiration!
