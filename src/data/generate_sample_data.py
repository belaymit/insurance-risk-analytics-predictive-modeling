"""
Generate sample insurance data for EDA analysis.
This script creates a realistic insurance dataset with relevant features.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_insurance_data(n_records=10000):
    """
    Generate sample insurance data with realistic distributions.
    
    Parameters:
    n_records (int): Number of records to generate
    
    Returns:
    pd.DataFrame: Generated insurance dataset
    """
    
    # Define data ranges and categories
    provinces = ['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba', 
                'Saskatchewan', 'Nova Scotia', 'New Brunswick', 'Newfoundland']
    
    vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Truck', 'Van', 'Convertible']
    
    vehicle_makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 
                    'Audi', 'Nissan', 'Hyundai', 'Volkswagen', 'Kia', 'Mazda']
    
    genders = ['Male', 'Female', 'Other']
    
    cover_types = ['Basic', 'Premium', 'Deluxe']
    
    # Generate base data
    data = []
    
    # Start date for 18-month period
    start_date = datetime(2022, 1, 1)
    
    for i in range(n_records):
        # Customer demographics
        age = np.random.normal(40, 12)
        age = max(18, min(80, age))  # Constrain age between 18-80
        
        gender = np.random.choice(genders, p=[0.48, 0.50, 0.02])
        province = np.random.choice(provinces, 
                                  p=[0.39, 0.23, 0.13, 0.12, 0.04, 0.03, 0.03, 0.02, 0.01])
        
        # Vehicle characteristics
        vehicle_type = np.random.choice(vehicle_types, 
                                      p=[0.35, 0.25, 0.15, 0.08, 0.08, 0.05, 0.04])
        vehicle_make = np.random.choice(vehicle_makes)
        vehicle_year = np.random.randint(2010, 2023)
        
        # Insurance details
        cover_type = np.random.choice(cover_types, p=[0.5, 0.35, 0.15])
        
        # Generate postal code (first 3 characters)
        postal_code = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(0,9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
        zip_code = random.randint(10000, 99999)
        
        # Financial calculations with realistic relationships
        base_premium = 800
        
        # Age factor (younger and very old drivers pay more)
        if age < 25:
            age_factor = 1.5
        elif age > 65:
            age_factor = 1.2
        else:
            age_factor = 1.0
            
        # Gender factor (slight difference)
        gender_factor = 1.1 if gender == 'Male' else 1.0
        
        # Vehicle type factor
        vehicle_factors = {'Sedan': 1.0, 'SUV': 1.1, 'Hatchback': 0.95, 
                          'Coupe': 1.3, 'Truck': 1.2, 'Van': 1.1, 'Convertible': 1.4}
        vehicle_factor = vehicle_factors[vehicle_type]
        
        # Province factor (different risk levels)
        province_factors = {'Ontario': 1.2, 'Quebec': 1.0, 'British Columbia': 1.1,
                           'Alberta': 1.05, 'Manitoba': 0.9, 'Saskatchewan': 0.85,
                           'Nova Scotia': 0.95, 'New Brunswick': 0.9, 'Newfoundland': 0.8}
        province_factor = province_factors[province]
        
        # Cover type factor
        cover_factors = {'Basic': 0.8, 'Premium': 1.0, 'Deluxe': 1.3}
        cover_factor = cover_factors[cover_type]
        
        # Calculate premium
        total_premium = base_premium * age_factor * gender_factor * vehicle_factor * province_factor * cover_factor
        total_premium *= np.random.uniform(0.8, 1.2)  # Add some randomness
        total_premium = round(total_premium, 2)
        
        # Vehicle value estimate
        vehicle_age = 2023 - vehicle_year
        base_value = 25000 if vehicle_type in ['SUV', 'Truck'] else 20000
        depreciation = min(0.15 * vehicle_age, 0.85)  # Cap depreciation at 85%
        custom_value_estimate = base_value * (1 - depreciation) * np.random.uniform(0.8, 1.2)
        custom_value_estimate = max(round(custom_value_estimate, 2), 1000)  # Minimum value of $1000
        
        # Claims generation (not everyone has claims)
        has_claim = np.random.random() < 0.15  # 15% claim probability
        
        if has_claim:
            # Claim amount influenced by vehicle value and type
            claim_severity = np.random.exponential(3000)
            claim_severity = min(claim_severity, custom_value_estimate * 0.8)  # Cap at 80% of vehicle value
            total_claims = max(round(claim_severity, 2), 100)  # Minimum claim of $100
        else:
            total_claims = 0.0
            
        # Random date within 18-month period
        random_days = np.random.randint(0, 547)  # 18 months ≈ 547 days
        transaction_date = start_date + timedelta(days=random_days)
        
        # Create record
        record = {
            'PolicyID': f'POL{1000000 + i}',
            'Province': province,
            'PostalCode': postal_code,
            'ZipCode': zip_code,
            'Gender': gender,
            'Age': round(age, 1),
            'VehicleType': vehicle_type,
            'VehicleMake': vehicle_make,
            'VehicleYear': vehicle_year,
            'CoverType': cover_type,
            'TotalPremium': total_premium,
            'TotalClaims': total_claims,
            'CustomValueEstimate': custom_value_estimate,
            'TransactionMonth': transaction_date.strftime('%Y-%m'),
            'TransactionDate': transaction_date.strftime('%Y-%m-%d')
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some calculated fields
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    df['LossRatio'] = df['LossRatio'].fillna(0)  # Fill inf values with 0
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    df['VehicleAge'] = 2023 - df['VehicleYear']
    df['ClaimFrequency'] = df.groupby('PolicyID')['HasClaim'].transform('sum')
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    print("Generating sample insurance dataset...")
    df = generate_insurance_data(10000)
    
    # Save to CSV
    df.to_csv('../../data/raw/insurance_data.csv', index=False)
    print(f"Dataset saved with {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info()) 