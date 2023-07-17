# Supervised Learning: From Scratch!
This repository features a collection of supervised learning algorithms for classification tasks implemented from scratch in Python. The algorithms are implemented in a Jupyter Notebook format, allowing for easy visualization of the results. A question and answer section is also provided to explain the inner workings of each algorithm, located in the `docs` folder.

## Algorithms
The following algorithms are implemented:
- K-Nearest Neighbors
- Logistic Regression
- Iterative Dichotomiser 3 (ID3)
- Decision Tree
The implementation of each algorithm is located in the `src/Models` folder.

## How to Use
To use the algorithms on your own custom dataset, simply head over to each represented notebooks
located in the `src` folder, and scroll down to the `External Dataset Evaluation` section located all the way on the bottom of the notebook. From there, you can specify the path to your dataset, and the notebook will automatically load the data and evaluate the algorithms on your dataset.  
  
<img width="500" alt="image" src="https://github.com/AlifioDitya/Supervised-Learning-From-Scratch/assets/103266159/25010f68-4c5b-4417-aad1-b4064217c1bb">

## Assumptions
The algorithms are implemented with the following assumptions:
- The dataset is in a CSV format
- For KNN, Logistic Regression, all data must be numerical and of no ordinality
- For ID3, the data must be an encoded form of a fully categorical dataset
- For Decision Tree, the data can be a mix of numerical and encoded form of categorical data
- Logistic Regression assumes the task of binary classification, since the binary cross-entropy loss function is used in implementation

## Requirements
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)