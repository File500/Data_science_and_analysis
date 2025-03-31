import pandas as pd
import numpy as np
import random


# Function to load the synthetic people dataset
def load_people_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading people data: {e}")


# Function to load the car listings dataset
def load_car_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading car data: {e}")



# Function to decide if a person would buy a car based on their profile
def would_buy_car(person):
    # Define probabilities based on income, credit score, and other factors

    # Base probability adjusted for income (higher income = higher probability)
    income_factor = min(person['Annual_Income'] / 100000, 1.0) * 0.4

    # Credit score factor (higher score = higher probability)
    credit_factor = (person['Credit_Score'] - 500) / 350 * 0.3

    # Years of employment factor (more stable employment = higher probability)
    employment_factor = min(person['Years_of_Employment'] / 10, 1.0) * 0.15

    # Finance status factor
    finance_map = {
        'Excellent': 0.15,
        'Good': 0.12,
        'Stable': 0.1,
        'Fair': 0.05,
        'Unstable': 0.02,
        'Poor': 0.01
    }
    finance_factor = finance_map.get(person['Finance_Status'], 0.05)

    # Children factor (more children might reduce probability)
    if person['Number_of_Children'] == -1:  # No information
        children_factor = 0
    else:
        children_factor = max(0, 0.05 - person['Number_of_Children'] * 0.01)

    # Calculate total probability
    total_probability = income_factor + credit_factor + employment_factor + finance_factor + children_factor

    # Cap probability at 90% (some randomness remains)
    buy_probability = min(total_probability, 0.9)

    return random.random() < buy_probability


# Function to match a car to a person
def match_car_to_person(person, car_data):
    # Higher income -> more expensive cars
    income_tier = person['Annual_Income'] / 20000  # Rough income tier

    # Filter cars based on rough affordability (people tend to buy cars worth about 1/3 of annual income)
    affordable_price = person['Annual_Income'] / 3

    # Add some randomness to price range
    price_range_min = affordable_price * 0.5
    price_range_max = affordable_price * 1.5

    # Filter cars in that price range (if we have enough data)
    affordable_cars = car_data[
        (car_data['selling_price'] >= price_range_min / 1000) &
        (car_data['selling_price'] <= price_range_max / 1000)
        ]

    # If no cars in range, just take all cars
    if len(affordable_cars) == 0:
        affordable_cars = car_data

    # Randomly select a car from the appropriate price range
    return affordable_cars.sample(1).iloc[0].to_dict()


# Main function to create the matched dataset
def create_car_matching_dataset(people_data_path, car_data_path, output_path=None):
    # Load datasets
    people_df = load_people_data(people_data_path)
    car_df = load_car_data(car_data_path)

    # Create empty dataframe for the matched car data
    matched_cars = []

    # Process each person
    for idx, person in people_df.iterrows():
        if would_buy_car(person):
            # Person buys a car - match them with an appropriate one
            car = match_car_to_person(person, car_df)
            car['person_id'] = idx  # Link to the person
            matched_cars.append(car)
        else:
            # Person doesn't buy a car - add NaN values
            nan_car = {col: np.nan for col in car_df.columns}
            nan_car['person_id'] = idx  # Link to the person
            matched_cars.append(nan_car)

    # Convert to DataFrame
    matched_cars_df = pd.DataFrame(matched_cars)

    # Save to CSV if output path is provided
    if output_path:
        matched_cars_df.to_csv(output_path, index=False)

    # Print statistics
    buy_count = matched_cars_df['name'].notna().sum()
    print(f"Generated {len(matched_cars_df)} car matching entries.")
    print(f"{buy_count} people ({buy_count / len(matched_cars_df) * 100:.1f}%) would buy a car.")
    print(f"{len(matched_cars_df) - buy_count} people would not buy a car (NaN values).")

    return matched_cars_df


# Example usage
if __name__ == "__main__":
    # Paths to your data files
    people_data_path = "../data/synthetic_people_data.csv"
    car_data_path = "../../data/clean_data/clean_car_price.csv"
    output_path = "../data/matched_cars_dataset.csv"

    # Create the matched dataset
    matched_cars_df = create_car_matching_dataset(people_data_path, car_data_path, output_path)

    # Display sample of the matched dataset
    print("\nSample of the matched dataset:")
    print(matched_cars_df.head())