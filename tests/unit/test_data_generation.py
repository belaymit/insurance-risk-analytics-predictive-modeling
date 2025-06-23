"""
Test suite for insurance data generation and validation.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_data_file_exists():
    """Test that the insurance data file or DVC pointer exists."""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'insurance_data.csv')
    dvc_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'insurance_data.csv.dvc')
    
    # In CI environment, we might only have DVC pointer file
    assert os.path.exists(data_path) or os.path.exists(dvc_path), \
        "Neither insurance data file nor DVC pointer exists"

def test_data_structure():
    """Test the structure and content of the insurance dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'insurance_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Test basic structure
        assert len(df) > 0, "Dataset is empty"
        assert len(df.columns) >= 15, "Dataset has too few columns"
        
        # Test required columns
        required_columns = ['PolicyID', 'Province', 'Gender', 'VehicleType', 
                          'TotalPremium', 'TotalClaims', 'CustomValueEstimate']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Test data types
        assert df['TotalPremium'].dtype in ['float64', 'int64'], "TotalPremium should be numeric"
        assert df['TotalClaims'].dtype in ['float64', 'int64'], "TotalClaims should be numeric"
        
        # Test data ranges
        assert df['TotalPremium'].min() >= 0, "TotalPremium should be non-negative"
        assert df['TotalClaims'].min() >= 0, "TotalClaims should be non-negative"
        
        print(f"✅ Data validation passed: {len(df)} records with {len(df.columns)} columns")
    else:
        print("ℹ️ Data file not available (likely DVC-managed), skipping structure tests")

def test_loss_ratio_calculation():
    """Test loss ratio calculations."""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'insurance_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Calculate loss ratio
        total_claims = df['TotalClaims'].sum()
        total_premiums = df['TotalPremium'].sum()
        loss_ratio = total_claims / total_premiums
        
        # Loss ratio should be between 0 and 2 (200%) for realistic data
        assert 0 <= loss_ratio <= 2, f"Loss ratio {loss_ratio} is outside realistic range"
        
        print(f"✅ Loss ratio validation passed: {loss_ratio:.4f}")
    else:
        print("ℹ️ Data file not available (likely DVC-managed), skipping loss ratio tests")

def test_data_completeness():
    """Test for missing values and data completeness."""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'insurance_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Check for missing values in critical columns
        critical_columns = ['TotalPremium', 'TotalClaims', 'Province', 'VehicleType']
        for col in critical_columns:
            missing_count = df[col].isnull().sum()
            assert missing_count == 0, f"Missing values found in critical column {col}: {missing_count}"
        
        print("✅ Data completeness validation passed")
    else:
        print("ℹ️ Data file not available (likely DVC-managed), skipping completeness tests")

if __name__ == "__main__":
    test_data_file_exists()
    test_data_structure()
    test_loss_ratio_calculation()
    test_data_completeness()
    print("🎉 All tests passed!") 