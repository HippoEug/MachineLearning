# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv') # Importing Data.csv
X = dataset.iloc[:, :-1].values # Taking all the rows and columns except the last one
y = dataset.iloc[:, 3].values # Taking the rows of the third column

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # Object imputer with parameters
imputer = imputer.fit(X[:, 1:3]) # Fit the missing data in all rows, and columns 1 and 2
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Replace missing data by the mean of the column
