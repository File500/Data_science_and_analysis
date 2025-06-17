import pandas as pd
import numpy as np
import random


# Function to generate synthetic people data
def generate_synthetic_people(num_samples=100):
    # Common occupations with approximate distributions
    occupations = {
        "Software Engineer": 8,
        "Teacher": 7,
        "Nurse": 6,
        "Doctor": 5,
        "Accountant": 5,
        "Marketing Manager": 5,
        "Graphic Designer": 4,
        "Sales Representative": 4,
        "Web Developer": 4,
        "Chef": 4,
        "Electrician": 3,
        "Physical Therapist": 3,
        "Data Scientist": 3,
        "Financial Analyst": 3,
        "HR Manager": 3,
        "Architect": 3,
        "Business Analyst": 3,
        "Project Manager": 3,
        "Lawyer": 2,
        "Dental Hygienist": 2,
        "Pharmacist": 2,
        "Veterinarian": 2,
        "Real Estate Agent": 2,
        "Insurance Agent": 2,
        "Executive Assistant": 2,
        "Customer Service Rep": 2,
        "Mechanical Engineer": 2,
        "IT Manager": 2,
        "Social Worker": 1,
        "Musician": 1,
        "Writer": 1,
        "Photographer": 1,
        "Personal Trainer": 1,
        "Interior Designer": 1,
    }

    # Weighted occupations list
    occupation_list = []
    for occupation, weight in occupations.items():
        occupation_list.extend([occupation] * weight)

    # Financial statuses
    finance_statuses = ["Stable"] * 70 + ["Unstable"] * 20 + ["Good"] * 4 + ["Fair"] * 3 + ["Excellent"] * 2 + [
        "Poor"] * 1

    # Number of children distribution
    num_children_options = [-1, 0, 1, 2, 3, 4]
    num_children_weights = [10, 25, 30, 25, 8, 2]  # Weighted probabilities

    # Create empty dataframe
    synthetic_people = []

    for i in range(num_samples):
        # Generate random values for each column
        occupation = random.choice(occupation_list)

        # Income ranges based on occupation (approximate ranges in thousands)
        income_ranges = {
            "Software Engineer": (70000, 150000),
            "Teacher": (40000, 70000),
            "Nurse": (50000, 90000),
            "Doctor": (120000, 300000),
            "Accountant": (50000, 90000),
            "Marketing Manager": (60000, 120000),
            "Graphic Designer": (40000, 80000),
            "Sales Representative": (35000, 80000),
            "Web Developer": (60000, 120000),
            "Chef": (35000, 70000),
            "Electrician": (40000, 80000),
            "Physical Therapist": (70000, 100000),
            "Data Scientist": (80000, 140000),
            "Financial Analyst": (60000, 110000),
            "HR Manager": (60000, 110000),
            "Architect": (70000, 120000),
            "Business Analyst": (60000, 100000),
            "Project Manager": (70000, 130000),
            "Lawyer": (80000, 200000),
            "Dental Hygienist": (60000, 90000),
            "Pharmacist": (100000, 140000),
            "Veterinarian": (80000, 120000),
            "Real Estate Agent": (50000, 120000),
            "Insurance Agent": (40000, 80000),
            "Executive Assistant": (45000, 70000),
            "Customer Service Rep": (30000, 50000),
            "Mechanical Engineer": (70000, 110000),
            "IT Manager": (90000, 140000),
            "Social Worker": (40000, 65000),
            "Musician": (25000, 80000),
            "Writer": (30000, 70000),
            "Photographer": (30000, 70000),
            "Personal Trainer": (30000, 60000),
            "Interior Designer": (40000, 80000),
        }

        # Default income range if occupation not in dictionary
        income_min, income_max = income_ranges.get(occupation, (30000, 80000))

        # Generate the values
        annual_income = round(random.uniform(income_min, income_max), 2)
        credit_score = random.randint(500, 850)
        years_employment = random.randint(1, 20)
        finance_status = random.choice(finance_statuses)
        num_children = random.choices(num_children_options, weights=num_children_weights, k=1)[0]

        # Create person entry
        person = {
            "Occupation": occupation,
            "Annual_Income": annual_income,
            "Credit_Score": credit_score,
            "Years_of_Employment": years_employment,
            "Finance_Status": finance_status,
            "Number_of_Children": num_children
        }

        synthetic_people.append(person)

    # Convert to DataFrame
    df = pd.DataFrame(synthetic_people)

    return df


# Generate the synthetic data
synthetic_df = generate_synthetic_people(100)

# Display the first few rows
print(synthetic_df.head())

# Save to CSV if needed
synthetic_df.to_csv('./data/synthetic_people_data.csv', index=True)

# Summary statistics
print("\nSummary Statistics:")
print(synthetic_df.describe())

# Value counts for categorical columns
print("\nOccupation Distribution:")
print(synthetic_df['Occupation'].value_counts().head(10))

print("\nFinance Status Distribution:")
print(synthetic_df['Finance_Status'].value_counts())

print("\nNumber of Children Distribution:")
print(synthetic_df['Number_of_Children'].value_counts())