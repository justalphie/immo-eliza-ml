import pickle
import numpy as np
import pandas as pd

import sklearn
from sklearn.linear_model import *
from sklearn.model_selection import cross_val_score

folder_name = input("What folder would you like to use? ")

def train_and_test_model(regressor: LinearRegression, X_train, y_train, X_test, y_test):
    regressor.fit(X_train, y_train)
    train_score = regressor.score(X_train, y_train)
    test_score = regressor.score(X_test, y_test)
    try:
        coeffs = list(zip(regressor.coef_.flatten().tolist(), regressor.feature_names_in_))
        coeffs.sort(key=lambda x: abs(x[0]), reverse=True)
        print(coeffs)
    except:
        pass
    print("Train score is: ", train_score)
    print("Test score is: ", test_score)
    return test_score


X_train = pd.read_csv(f"./{folder_name}/data/X_train.csv")
X_train.set_index("property_id", inplace=True)
X_test = pd.read_csv(f"./{folder_name}/data/X_test.csv")
X_test.set_index("property_id", inplace=True)


y_train = pd.read_csv(f"./{folder_name}/data/y_train.csv")
y_train.set_index("property_id", inplace=True)
y_test = pd.read_csv(f"./{folder_name}/data/y_test.csv")
y_test.set_index("property_id", inplace=True)


model = Ridge()
test_score = train_and_test_model(model, X_train, y_train, X_test, y_test)

with open(f"./{folder_name}/models/ridge.pickle", "wb") as f:
    pickle.dump(model, f)

with open(f"./{folder_name}/models/ridge.txt", "w") as f:
    f.write(str(test_score))