import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("../data/heart.csv")

# Split data into features X and result y
X = data.drop("target", axis=1)
y = data.target

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# choose RandomForestClassifier for that data set
rf_model = RandomForestClassifier()

# fit the model
rf_model.fit(X_train, y_train)

# get first score results
rf_model.score(X_test, y_test)

# Safe model as pickle file
pickle.dump(rf_model, open("rf_model_heart_disease.pkl", "wb"))
