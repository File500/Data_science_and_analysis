import pandas as pd
import numpy as np
from joblib import load
import warnings
import findspark
from pyspark.sql import SparkSession
from tabulate import tabulate
from difflib import SequenceMatcher
from tqdm import tqdm

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
    "name_match_w": 0.7028,
    "year_match_w": 0.6943,
    "selling_price_match_w": 0.88,
    "km_driven_match_w": 0.6614,
    "fuel_match_w": 0.8857,
    "seller_type_match_w": 0.7,
    "transmission_match_w": 0.74,
    "owner_match_w": 0.6714
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

    if ((row_original.isna().sum().sum() and not row_recommendations.isna().sum().sum())
            or (not row_original.isna().sum().sum() and row_recommendations.isna().sum().sum())):
        update_points(rec_type=rec_type, match_type="incorrect")
        return

    #print(row_original.iloc[0])
    #print( row_recommendations.iloc[0])

    for col in row_recommendations.columns:
        if col != "person_id":

            match_name = col + "_match"
            int_threshold = 2
            float_threshold = 1000
            str_threshold = 0.8

            if row_recommendations[col].dtype == np.integer:

                int_diff = row_original.iloc[0][col] - row_recommendations.iloc[0][col]

                if int_diff == 0:
                    update_points(rec_type=rec_type, match_type="match", points_type=match_name)
                elif int_threshold > int_diff > -int_threshold:
                    update_points(rec_type=rec_type, match_type="similar", points_type=match_name)
                else:
                    update_points(rec_type=rec_type, match_type="different", points_type=match_name)

            elif row_recommendations[col].dtype == np.float32 or row_recommendations[col].dtype == np.float64:

                float_diff = row_original.iloc[0][col] - row_recommendations.iloc[0][col]

                if float_diff == 0:
                    update_points(rec_type=rec_type, match_type="match", points_type=match_name)
                elif float_threshold > float_diff > -float_threshold:
                    update_points(rec_type=rec_type, match_type="similar", points_type=match_name)
                else:
                    update_points(rec_type=rec_type, match_type="different", points_type=match_name)

            else:

                ratio = SequenceMatcher(None, row_original.iloc[0][col], row_recommendations.iloc[0][col]).ratio()

                if ratio == 1.0:
                    update_points(rec_type=rec_type, match_type="match", points_type=match_name)
                elif ratio >= str_threshold:
                    update_points(rec_type=rec_type, match_type="similar", points_type=match_name)
                else:
                    update_points(rec_type=rec_type, match_type="different", points_type=match_name)


def test_system(recommendations: pd.DataFrame, user_cars: pd.DataFrame, rec_type: str) -> pd.DataFrame:

    if rec_type != 'smart' and rec_type != 'naive':
        print('Invalid rec_type')
        return pd.DataFrame()

    print(f"Evaluating recommendations for {rec_type} system....")

    for index_user, data_user in tqdm(user_cars.iterrows()):
        rec_data = pd.DataFrame()

        if rec_type == 'smart':
            rec_data = recommendations.where(recommendations['person_id'] == index_user).dropna(axis=0).__deepcopy__()
        elif rec_type == 'naive':
            rec_data = recommendations.__deepcopy__()

        data_user_row = pd.DataFrame(data=[data_user], index=[index_user])

        for index_rec, data_rec in rec_data.iterrows():

            data_rec_row = pd.DataFrame(data=[data_rec], index=[index_rec])

            compare_rows(data_rec_row, data_user_row, rec_type)

    result_df = pd.DataFrame.from_dict(systems[rec_type], orient='index')
    print(result_df.columns)
    return result_df


findspark.init()
spark = SparkSession.builder.appName("Price").getOrCreate()

car_list = spark.read.csv(path="../data/clean_data/clean_car_price.csv", header=True, inferSchema=True)
car_list.createOrReplaceTempView("price")

naive_car_recommendation = find_car_listing_naive()
smart_car_recommendation = pd.read_csv("./data/recommendations.csv")

mock_matched_cars = pd.read_csv("./data/matched_cars_dataset.csv")
mock_matched_cars = mock_matched_cars.drop(columns=['Unnamed: 0'])
mock_matched_cars['year'] = mock_matched_cars['year'].apply(lambda x: int(x) if not pd.isna(x) else x)

test_results_smart = test_system(smart_car_recommendation, mock_matched_cars, "smart")
test_results_smart.to_csv(path_or_buf="./test_results_smart.csv")

test_results_naive = test_system(naive_car_recommendation, mock_matched_cars, "naive")
test_results_naive.to_csv(path_or_buf="./test_results_naive.csv")

spark.stop()
