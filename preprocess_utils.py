import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, KBinsDiscretizer

def remove_rare_sale_types(df):
    index_type_of_sale = df[(df["type_of_sale"] == "PUBLIC_SALE")| (df["type_of_sale"] == "LIFE_ANNUITY")].index
    df.drop(index_type_of_sale, inplace=True)

def replace_incorrect_kitchen(df):
    df.replace({"kitchen_type":"USA_UNINSTALLED"}, 'USA_HYPER_EQUIPPED', inplace=True)

def clean_dataset(df):
    df.set_index('property_id', inplace=True)
    remove_rare_sale_types(df)
    replace_incorrect_kitchen(df)
    df = df[['price', 'property_subtype',  'number_of_rooms', 'living_area', "surface_of_good", 'kitchen_type', 'garden', 'garden_area', 'furnished', 'open_fire', 'terrace','number_of_facades', 'state_of_building', 'postal_code', "latitude", "longitude"]]
    return df

def concat_latlon(lat, lon):
    return str(int(lat))+"x"+str(int(lon))

# TODO
def get_initial_preprocessings():
    return {

        "imputations": {
            # most frequent
            'property_subtype': SimpleImputer(strategy="most_frequent"),
            'number_of_rooms': SimpleImputer(strategy='most_frequent'),
            'kitchen_type': SimpleImputer(strategy='most_frequent'),
            'number_of_facades': SimpleImputer(strategy='most_frequent'),
            'state_of_building': SimpleImputer(strategy='most_frequent'),
            "postal_code": SimpleImputer(strategy='most_frequent'),
            "latitude": SimpleImputer(strategy='most_frequent'),
            "longitude": SimpleImputer(strategy='most_frequent'),
            # mean
            'living_area': SimpleImputer(strategy="mean"),
            # constant
            "garden": SimpleImputer(strategy='constant', fill_value=0.0), 
            'garden_area': SimpleImputer(strategy='constant', fill_value=0.0),
            'surface_of_good': SimpleImputer(strategy="constant", fill_value=0.0),
            "terrace": SimpleImputer(strategy='constant', fill_value=0.0), 
            "open_fire": SimpleImputer(strategy='constant', fill_value=0), 
            "furnished": SimpleImputer(strategy='constant', fill_value=0.0),
        },

        "encodings": {
            'property_subtype' : OneHotEncoder(sparse_output=False, handle_unknown='ignore'), 
            'number_of_rooms' : OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'number_of_facades' : OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'kitchen_type' : OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'state_of_building' : OneHotEncoder(sparse_output=False, handle_unknown='ignore')},

        "normalizations" : {
            # normalize
            'living_area': MinMaxScaler(),
            'surface_of_good': MinMaxScaler(),
            # standardize
            'garden_area': StandardScaler()
        },

        "binnings" : {
            "postal_code": KBinsDiscretizer(25, strategy="quantile"),
            "latitude": KBinsDiscretizer(25, strategy="quantile", encode="ordinal"),
            "longitude": KBinsDiscretizer(25, strategy="quantile", encode="ordinal")},

        "encoding_latlon" : {
            'latlon' : OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        }
        
    }

def apply_preprocessings(preprocessings, X_train=None, X_test=None):
    # Apply the imputation on the train and test datasets
    for column_to_impute, imputer in preprocessings['imputations'].items():
        if X_train is not None:
            X_train[[column_to_impute]] = imputer.fit_transform(X_train[[column_to_impute]])
        if X_test is not None:
            X_test[[column_to_impute]] = imputer.transform(X_test[[column_to_impute]])
    #Apply OneHotEncoder on the train and test datasets
    for column_to_encode, encoder in preprocessings["encodings"].items():
        # fit_transform X_train
        if X_train is not None:
            transformed_X_train = encoder.fit_transform(X_train[[column_to_encode]])
            transformed_X_train_df = pd.DataFrame(transformed_X_train, index=X_train.index, columns=encoder.get_feature_names_out())
            X_train = pd.concat([X_train, transformed_X_train_df], axis=1).drop([column_to_encode], axis=1)
        # transform X_test
        if X_test is not None:
            transformed_X_test = encoder.transform(X_test[[column_to_encode]])
            transformed_X_test_df = pd.DataFrame(transformed_X_test, index=X_test.index, columns=encoder.get_feature_names_out())
            X_test = pd.concat([X_test, transformed_X_test_df], axis=1).drop([column_to_encode], axis=1)
    # apply normalization and standardization on train and test sets
    for column_to_normalize, normalizer in preprocessings["normalizations"].items():
        if X_train is not None:
            X_train[[column_to_normalize]] = normalizer.fit_transform(X_train[[column_to_normalize]])
        if X_test is not None:    
            X_test[[column_to_normalize]] = normalizer.transform(X_test[[column_to_normalize]])
    # apply binning and encoding the bins with OneHotEncoder
    for column_to_bin, discretizer in preprocessings["binnings"].items():
        if X_train is not None:
            if discretizer.get_params()['encode'] == 'onehot':
                result = discretizer.fit_transform(X_train[[column_to_bin]]).toarray()
                transformed_X_train_df = pd.DataFrame(result, index=X_train.index, columns=discretizer.get_feature_names_out())
                X_train = pd.concat([X_train, transformed_X_train_df], axis=1).drop([column_to_bin], axis=1)
            else:
                result = discretizer.fit_transform(X_train[[column_to_bin]])
                transformed_X_train_df = pd.DataFrame(result, index=X_train.index, columns=[column_to_bin])
                X_train[[column_to_bin]] = transformed_X_train_df
        if X_test is not None:
            if discretizer.get_params()['encode'] == 'onehot':
                result = discretizer.transform(X_test[[column_to_bin]]).toarray()
                transformed_X_test_df = pd.DataFrame(result, index=X_test.index, columns=discretizer.get_feature_names_out())
                X_test = pd.concat([X_test, transformed_X_test_df], axis=1).drop([column_to_bin], axis=1)
            else:
                result = discretizer.transform(X_test[[column_to_bin]])
                transformed_X_test_df = pd.DataFrame(result, index=X_test.index, columns=[column_to_bin])
                X_test[[column_to_bin]] = transformed_X_test_df 
    #adding the latitude+longitude column
    if X_train is not None:
        X_train["latlon"] = X_train.apply(lambda row: concat_latlon(row["latitude"], row["longitude"]), axis=1)
        X_train.drop(["latitude","longitude"], axis=1, inplace=True)
    if X_test is not None:
        X_test["latlon"] = X_test.apply(lambda row: concat_latlon(row["latitude"], row["longitude"]), axis=1)
        X_test.drop(["latitude", "longitude"], axis=1, inplace=True)
    #Encoding the latlon column with OneHotEncoder
    for column_to_encode, encoder in preprocessings["encoding_latlon"].items():
        if X_train is not None:
            # fit_transform X_train
            transformed_X_train = encoder.fit_transform(X_train[[column_to_encode]])
            transformed_X_train_df = pd.DataFrame(transformed_X_train, index=X_train.index, columns=encoder.get_feature_names_out())
            X_train = pd.concat([X_train, transformed_X_train_df], axis=1).drop([column_to_encode], axis=1)
        if X_test is not None:
            # transform X_test
            transformed_X_test = encoder.transform(X_test[[column_to_encode]])
            transformed_X_test_df = pd.DataFrame(transformed_X_test, index=X_test.index, columns=encoder.get_feature_names_out())
            X_test = pd.concat([X_test, transformed_X_test_df], axis=1).drop([column_to_encode], axis=1)
        #new preprocessing step: multiply columns with living_area
        #(because these columns change the price per square meter)
        if X_train is not None:
            for column in X_train.columns:
                if "postal_code" in column or "latlon" in column or "property_subtype" in column or "state_of_building" in column:
                    X_train[column] = X_train[column] * X_train["living_area"]
        if X_test is not None:
            for column in X_test.columns:
                if "postal_code" in column or "latlon" in column or "property_subtype" in column or "state_of_building" in column:
                    X_test[column] = X_test[column] * X_test["living_area"]

    # return the updated sets
    return X_train, X_test

