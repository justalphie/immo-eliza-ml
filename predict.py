import pickle
import pandas as pd
import sklearn
from preprocess_utils import *

folder_name = input("Indicate the folder, please. ")

# load new dataset.csv (without price)

df = pd.read_csv(f"./{folder_name}/data/dataset.csv")

#cleaning the dataset
df = clean_dataset(df)

X_test = df.iloc[:, 1:]
y_test = df["price"]

with open(f"./{folder_name}/preprocessings/preprocessings.pickle", "rb") as f:
    preprocessings = pickle.load(f)

X_train, X_test = apply_preprocessings(preprocessings, X_train=None, X_test=X_test)


# load a model 
with open (f"./{folder_name}/models/ridge.pickle", "rb") as f:
    ridge = pickle.load(f)

#predict the y using the model 
y_pred = ridge.predict(X_test)

#save csv predictions
y_pred_df = pd.DataFrame(y_pred, index=X_test.index, columns=["price_pred"])
y_pred_df.to_csv(f"./{folder_name}/data/y_predict.csv")
