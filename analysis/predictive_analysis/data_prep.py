import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
import joblib
from datetime import datetime

if (os.path.exists("../../data/train_test_validate_data/train_test_data.csv")):
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
data_reg.to_csv("regg_data.csv")