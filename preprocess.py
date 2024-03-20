import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess_utils import *

#working on houses or apartments?
folder_name = input("Which folder do you want to use? ")

df = pd.read_csv(f"./{folder_name}/data/dataset.csv")

#cleaning the dataset
df = clean_dataset(df)

#splitting to train and test
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['price'], test_size=0.2, random_state=40)

# getting the parameters of preprocessing
preprocessings = get_initial_preprocessings()

# preprocessing the train and test sets
X_train, X_test = apply_preprocessings(preprocessings, X_train=X_train, X_test=X_test)

  
#saving the files
X_train.to_csv(f"./{folder_name}/data/X_train.csv")
X_test.to_csv(f"./{folder_name}/data/X_test.csv")
y_train.to_csv(f"./{folder_name}/data/y_train.csv")
y_test.to_csv(f"./{folder_name}/data/y_test.csv")

with open(f"./{folder_name}/preprocessings/preprocessings.pickle", "wb") as f:
    pickle.dump(preprocessings, f)
