import pandas as pd
import numpy as np
from joblib import load
import warnings
import findspark
from pyspark.sql import SparkSession
from tabulate import tabulate


def warn(*args, **kwargs):
    pass


def sql_to_df(command: str) -> pd.DataFrame:
    return spark.sql(command).toPandas()


def find_car_listing(customer_price: float) -> pd.DataFrame:
    threshold = 1000

    upper_bound = customer_price + threshold
    lower_bound = customer_price - threshold

    car_recommendation = sql_to_df(f'''
        SELECT * FROM price 
        WHERE selling_price <= {upper_bound} 
        AND selling_price >= {lower_bound}
        AND owner NOT LIKE '%Test Drive Car%'
        ORDER BY year DESC, km_driven ASC 
        LIMIT 5
    ''')

    car_recommendation = car_recommendation.drop(columns=['_c0'])

    return car_recommendation


def prepare_features(df, label_mapping_dict, prep_ptype) -> pd.DataFrame:
    df = df.copy()
    df['Annual_Income_Credit_Score'] = df['Annual_Income'] / df['Credit_Score']
    df['Annual_Income_Years_Employment'] = df['Annual_Income'] / df['Years_of_Employment']
    df['Years_Employment_Credit_Score'] = df['Credit_Score'] / df['Years_of_Employment']
    df['Finance_Status'] = df['Finance_Status'].replace(label_mapping_dict)

    if prep_ptype == 'class':
        return pd.get_dummies(df, drop_first=True, columns=['Occupation'], dtype=float)

    elif prep_ptype == 'regg':
        return df.drop(columns=['Occupation'])

    return pd.DataFrame()


findspark.init()
spark = SparkSession.builder.appName("Price").getOrCreate()

warnings.warn = warn
warnings.filterwarnings('ignore')

reg_model = load('../models/best_model_Gradient_Boosting_R2_0.8332_20250130_172538.joblib')
class_model = load('../models/best_model_XGBoost_ACC_0.9481_20250131_201426.joblib')
# reg_model = load('../models_retrained/gradient_boosting_model_20250521_165657.joblib')['model']
# class_model = load('../models_retrained/xgboost_binary_classifier_20250414_121504.joblib')['model']

label_mapping = {
    'Unknown': 0,
    'Poor': 1,
    'Unstable': 2,
    'Fair': 3,
    'Stable': 4,
    'Good': 5,
    'Excellent': 6
}

car_list = spark.read.csv(path="../data/clean_data/clean_car_price.csv", header=True, inferSchema=True)
car_list.createOrReplaceTempView("price")

data_test = pd.read_csv("data/synthetic_people_data.csv")
# data_test = data_test.where((data_test['Annual_Income'] < 45000)).dropna(axis=0)
output_csv = pd.DataFrame(columns=["name","year","selling_price","km_driven","fuel","seller_type","transmission","owner","person_id"])
threshold = 0.38

for index, data in data_test.iterrows():

    data_df = pd.DataFrame(data=[data], index=[index])
    output_customer_info = data_df.__deepcopy__()

    data_class_model = prepare_features(data_df, label_mapping, 'class')
    data_reg_model = prepare_features(data_df, label_mapping, 'regg')

    expected_columns = class_model.feature_names_in_
    missing_cols = set(expected_columns) - set(data_class_model.columns)

    for c in missing_cols:
        data_class_model[c] = 0
    data_class_model = data_class_model[expected_columns]

    class_prediction = class_model.predict(data_class_model)

    if class_model.predict_proba(data_class_model)[0,1] >= threshold:

        regg_prediction = reg_model.predict(data_reg_model)
        price_for_customer = regg_prediction[0].astype(int)
        recommendations = find_car_listing(price_for_customer)

        recommendations['person_id'] = index

        output_csv = pd.concat([output_csv, recommendations], ignore_index=True)

        print(f"\nRecommendations for customer: with index {index}\n")
        print(tabulate(output_customer_info, headers='keys', tablefmt='pretty', showindex=False), '\n')
        print(tabulate(recommendations, headers='keys', tablefmt='pretty', showindex=False), '\n')

    else:
        print('\nCustomer with likelihood of 95% will not buy a car at the moment. For customer:')
        print(tabulate(output_customer_info, headers='keys', tablefmt='pretty', showindex=False), '\n')

output_csv.to_csv("./data/recommendations.csv", index=False)
spark.stop()
