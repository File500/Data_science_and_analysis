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
        AND seller_type NOT LIKE '%Test Drive Car%'
        ORDER BY year DESC, km_driven ASC 
        LIMIT 5
    ''')

    car_recommendation = car_recommendation.drop(columns=['_c0'])

    return car_recommendation


findspark.init()
spark = SparkSession.builder.appName("Price").getOrCreate()

warnings.warn = warn
warnings.filterwarnings('ignore')

reg_model = load('../models/best_model_Gradient_Boosting_R2_0.8332_20250130_172538.joblib')
class_model = load('../models/best_model_XGBoost_ACC_0.9481_20250131_201426.joblib')

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

data_columns = ['Occupation', 'Annual_Income', 'Credit_Score', 'Years_of_Employment', 'Finance_Status',
                'Number_of_Children']
input_values = []

for col in data_columns:
    prompt_text = "Please specify your " + col.replace('_', ' ') + ':'
    x = np.nan

    if col == 'Occupation' or col == 'Finance_Status':
        x = str(input(prompt_text))
    else:
        x = int(input(prompt_text))
        x = float(x)

    input_values.append(x)

data = pd.DataFrame([input_values], columns=data_columns)
output_customer_info = data.__deepcopy__()

data['Annual_Income_Credit_Score'] = data['Annual_Income'] / data['Credit_Score']
data['Annual_Income_Years_Employment'] = data['Annual_Income'] / data['Years_of_Employment']
data['Years_Employment_Credit_Score'] = data['Credit_Score'] / data['Years_of_Employment']

data_class_model = data
data_reg_model = data.drop(columns=['Occupation'])

data_reg_model['Finance_Status'] = data_reg_model['Finance_Status'].replace(label_mapping)

train_data = pd.read_csv('../data/train_test_validate_data/train_test_data.csv')

train_data = train_data.drop(columns=['Price', 'Car', 'Unnamed: 0'])
train_data = train_data.where((train_data['Annual_Income'] < 45000)).dropna(axis=0)

train_data['Annual_Income_Credit_Score'] = train_data['Annual_Income'] / train_data['Credit_Score']
train_data['Annual_Income_Years_Employment'] = train_data['Annual_Income'] / train_data['Years_of_Employment']
train_data['Years_Employment_Credit_Score'] = train_data['Credit_Score'] / train_data['Years_of_Employment']

train_data.loc[len(train_data)] = data_class_model.iloc[0]

train_data['Finance_Status'] = train_data['Finance_Status'].replace(label_mapping)
train_data = pd.get_dummies(train_data, drop_first=True, columns=['Occupation'], dtype=float)

data_class_model = train_data.tail(1)

class_prediction = class_model.predict(data_class_model)

if class_prediction[0] == 1:

    regg_prediction = reg_model.predict(data_reg_model)
    price_for_customer = regg_prediction[0].astype(int)
    recommendations = find_car_listing(price_for_customer)

    print("\nRecommendations for customer:\n")
    print(tabulate(output_customer_info, headers='keys', tablefmt='pretty', showindex=False), '\n')
    print(tabulate(recommendations, headers='keys', tablefmt='pretty', showindex=False), '\n')

else:
    print('Customer with likelihood of 95% will not buy a car at the moment.')


spark.stop()
