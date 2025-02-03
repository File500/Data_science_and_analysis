import pandas as pd
import numpy as np
import warnings
import findspark
from pyspark.sql import SparkSession
from tabulate import tabulate


def warn(*args, **kwargs):
    pass


def sql_to_df(command: str) -> pd.DataFrame:
    return spark.sql(command).toPandas()


test_price = 20000

warnings.warn = warn
warnings.filterwarnings('ignore')

findspark.init()
spark = SparkSession.builder.appName("Price").getOrCreate()

car_list = spark.read.csv(path="../data/clean_data/clean_car_price.csv", header=True, inferSchema=True)
car_list.createOrReplaceTempView("price")

################################################
threshold = 1000

upper_bound = test_price + threshold
lower_bound = test_price - threshold

result = sql_to_df(f'''
    SELECT * FROM price 
    WHERE selling_price <= {upper_bound} 
    AND selling_price >= {lower_bound}
    AND owner NOT LIKE '%Test Drive Car%'
    ORDER BY year DESC, km_driven ASC 
    LIMIT 5
''')

result = result.drop(columns=['_c0'])

print("\nMatching cars found:")
print(tabulate(result, headers='keys', tablefmt='pretty', showindex=False), '\n')

################################################

spark.stop()
