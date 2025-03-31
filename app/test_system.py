import pandas as pd
import numpy as np
from joblib import load
import warnings
import findspark
from pyspark.sql import SparkSession
from tabulate import tabulate

def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings('ignore')

match_values = {
    "match": 1,
    "similar": 0.5,
    "different": 0,
    "incorrect": -1
}

match_weights = {
    "name_match_w": 0.9,
    "year_match_w": 1,
    "selling_price_match_w": 1,
    "km_driven_match_w": 1,
    "fuel_match_w": 0.7,
    "seller_type_match_w": 0.7,
    "transmission_match_w": 0.7,
    "owner_match_w": 0.7
}

points_smart = {
    "name_match": 0,
    "year_match": 0,
    "selling_price_match": 0,
    "km_driven_match": 0,
    "fuel_match": 0,
    "seller_type_match": 0,
    "transmission_match": 0,
    "owner_match": 0
}

points_naive = {
    "name_match": 0,
    "year_match": 0,
    "selling_price_match": 0,
    "km_driven_match": 0,
    "fuel_match": 0,
    "seller_type_match": 0,
    "transmission_match": 0,
    "owner_match": 0
}

systems = {
    "smart": points_smart,
    "naive": points_naive
}

def warn(*args, **kwargs):
    pass


def sql_to_df(command: str) -> pd.DataFrame:
    return spark.sql(command).toPandas()

def find_car_listing_naive() -> pd.DataFrame:

    car_recommendation = sql_to_df(f'''
        SELECT * FROM price 
        WHERE owner NOT LIKE '%Test Drive Car%'
        ORDER BY year DESC, km_driven ASC 
        LIMIT 5
    ''')

    car_recommendation = car_recommendation.drop(columns=['_c0'])

    return car_recommendation


def update_points(rec_type: str, match_type: str, points_type: str = "all") -> None:

    if points_type == "all":
        for key in systems[rec_type].keys():
            weights_type = key + "_w"
            systems[rec_type][key] += match_values[match_type] * match_weights[weights_type]
    else:
        weights_type = points_type + "_w"
        systems[rec_type][points_type] += match_values[match_type] * match_weights[weights_type]


def compare_rows(row_original: pd.DataFrame, row_recommendations: pd.DataFrame, rec_type: str) -> None:

    if (row_original.empty and not row_recommendations.empty) or (not row_original.empty and row_recommendations.empty):
        update_points(rec_type=rec_type, match_type="incorrect")
        return

    for col in row_recommendations.columns:

def test_system(recommendations: pd.DataFrame, user_cars: pd.DataFrame, rec_type: str) -> pd.DataFrame:

    if rec_type != 'smart' and rec_type != 'naive':
        print('Invalid rec_type')
        return pd.DataFrame()

    for index_user, data_user in user_cars.iterrows():
        rec_data = pd.DataFrame()

        if rec_type == 'smart':
            rec_data = recommendations.where(recommendations['person_id'] == index_user).dropna(axis=0).__deepcopy__()
        elif rec_type == 'naive':
            rec_data = recommendations.__deepcopy__()

        data_user_row = pd.DataFrame(data=[data_user], index=[index_user])

        for index_rec, data_rec in rec_data.iterrows():

            data_rec_row = pd.DataFrame(data=[data_rec], index=[index_rec])
            compare_rows(data_rec_row, data_user_row, rec_type)

    return pd.DataFrame()


findspark.init()
spark = SparkSession.builder.appName("Price").getOrCreate()

car_list = spark.read.csv(path="../data/clean_data/clean_car_price.csv", header=True, inferSchema=True)
car_list.createOrReplaceTempView("price")

naive_car_recommendation = find_car_listing_naive()
smart_car_recommendation = pd.read_csv("./data/recommendations.csv")
mock_matched_cars = pd.read_csv("./data/matched_cars_dataset.csv")
mock_matched_cars =  mock_matched_cars.drop(columns=['Unnamed: 0'])

test_results_smart = test_system(smart_car_recommendation, mock_matched_cars, "smart")
test_results_smart.to_csv(path_or_buf="./test_results_smart.csv")

test_results_naive = test_system(naive_car_recommendation, mock_matched_cars, "naive")
test_results_naive.to_csv(path_or_buf="./test_results_naive.csv")

