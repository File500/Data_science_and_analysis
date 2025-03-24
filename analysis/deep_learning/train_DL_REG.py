def warn(*args, **kwargs):
    pass

import warnings

warnings.warn = warn

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
import joblib
from datetime import datetime


from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, make_scorer

class CarPriceRegressor(object):
    def __init__(self):
        self.model = None

def train_DL_REG(X_train, X_test, y_train, y_test):
    return 0


def main():

    if(os.path.exists("../../data/train_test_validate_data/train_test_data.csv")):
        data = pd.read_csv("../../data/train_test_validate_data/train_test_data.csv")
    else:
        ownership_df_class = pd.read_csv("../../data/final_data/final_ownership_data.csv")
        sales_df_reg = pd.read_csv("../../data/final_data/final_sales_data.csv")

        ownership_df_class = ownership_df_class.drop(columns=['Unnamed: 0', '_c0'])
        sales_df_reg = sales_df_reg.drop(columns=['Unnamed: 0', '_c0'])
        sales_df_reg = sales_df_reg[['Annual_Income', 'Price']]

        def find_closest_income_match(target_income, reference_df):
            closest_income_idx = (reference_df['Annual_Income'] - target_income).abs().idxmin()
            return reference_df.loc[closest_income_idx, 'Price']

        ownership_df_class['Price'] = ownership_df_class.apply(
            lambda row: find_closest_income_match(row['Annual_Income'], sales_df_reg)
            if row['Car'] == 'Yes'
            else 0,
            axis=1
        )

        ownership_sales = ownership_df_class.__deepcopy__()
        ownership_sales.to_csv(path_or_buf="../../data/train_test_validate_data/train_test_data.csv")
        data = pd.read_csv("../../data/train_test_validate_data/train_test_data.csv")

    data_reg = data.__deepcopy__()

    data_reg = data_reg.where(cond=(data_reg['Car'] == 'Yes')).dropna(axis=0).drop(columns=['Unnamed: 0', 'Car', 'Occupation'])

    label_mapping = {
        'Unknown': 0,
        'Unstable': 1,
        'Fair': 2,
        'Stable': 3,
        'Good': 4,
        'Excellent': 5
    }

    data_reg['Finance_Status'] = data_reg['Finance_Status'].replace(label_mapping)
    data_reg = data_reg.where((data_reg['Annual_Income'] < 45000)).dropna(axis=0)

    data_reg['Annual_Income_Credit_Score'] = data_reg['Annual_Income'] / data_reg['Credit_Score']
    data_reg['Annual_Income_Years_Employment'] = data_reg['Annual_Income'] / data_reg['Years_of_Employment']
    data_reg['Years_Employment_Credit_Score'] = data_reg['Credit_Score'] / data_reg['Years_of_Employment']

    Y = data_reg['Price']
    X = data_reg.drop(columns=['Price'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    train_DL_REG(X_train, y_train, X_test, y_test)



if __name__ == '__main__':
    main()
    print("Done")
