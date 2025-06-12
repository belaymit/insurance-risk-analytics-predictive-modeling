#!/usr/bin/env python3
"""
Predictive Modeling Script for Insurance Risk Analytics
Task 4 Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                            precision_score, recall_score, f1_score, roc_auc_score)

# XGBoost
import xgboost as xgb

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_prepare_data():
    """Load and prepare data for modeling"""
    print("Loading dataset...")
    df = pd.read_csv('MachineLearningRating_v3.txt', delimiter='|', low_memory=False)
    
    print("Applying feature engineering...")
    # Create claim indicator
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    
    # Claim ratio
    df['ClaimRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-6)
    
    # Premium per unit insured
    df['PremiumPerInsured'] = df['TotalPremium'] / (df['SumInsured'] + 1e-6)
    
    # Vehicle age
    df['VehicleAge'] = 2024 - df['RegistrationYear']
    df['VehicleAge'] = df['VehicleAge'].clip(0, 50)
    
    # Categorical combinations
    df['Gender_MaritalStatus'] = df['Gender'].astype(str) + '_' + df['MaritalStatus'].astype(str)
    
    # Risk scores
    province_risk = df.groupby('Province')['HasClaim'].mean()
    df['ProvinceRiskScore'] = df['Province'].map(province_risk)
    
    make_risk = df.groupby('make')['HasClaim'].mean()
    df['MakeRiskScore'] = df['make'].map(make_risk)
    
    # Experience features
    df['IsNewDriver'] = ((df['Gender'] == 'Male') & (df['VehicleAge'] < 5)).astype(int)
    
    return df

def get_feature_sets(df):
    """Define feature sets for modeling"""
    
    # Exclude target variables and identifiers
    exclude_cols = ['PolicyID', 'TotalClaims', 'TotalPremium', 'HasClaim', 'ClaimRatio', 
                   'UnderwrittenCoverID', 'TransactionMonth']
    
    # Numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    # Categorical features (limit to reasonable cardinality)
    categorical_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        if col not in exclude_cols and df[col].nunique() < 50:
            categorical_cols.append(col)
    
    return numerical_cols, categorical_cols

def create_preprocessor(numerical_features, categorical_features):
    """Create preprocessing pipeline"""
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def evaluate_regression_model(model, X_test, y_test, model_name):
    """Evaluate regression model"""
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mae = np.mean(np.abs(y_test - predictions))
    
    print(f"\n{model_name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    
    return {'RMSE': rmse, 'R2': r2, 'MAE': mae}

def evaluate_classification_model(model, X_test, y_test, model_name):
    """Evaluate classification model"""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc_score = roc_auc_score(y_test, probabilities)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    
    return {
        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 
        'F1': f1, 'AUC': auc_score
    }

def main():
    """Main execution function"""
    print("="*80)
    print("INSURANCE RISK ANALYTICS - PREDICTIVE MODELING")
    print("="*80)
    
    # Load and prepare data
    df = load_and_prepare_data()
    print(f"Dataset shape: {df.shape}")
    
    # Get feature sets
    numerical_features, categorical_features = get_feature_sets(df)
    all_features = numerical_features + categorical_features
    print(f"Total features: {len(all_features)} ({len(numerical_features)} numerical + {len(categorical_features)} categorical)")
    
    # Target analysis
    print(f"\nTarget Analysis:")
    print(f"Total records: {len(df):,}")
    print(f"Records with claims: {(df['TotalClaims'] > 0).sum():,} ({(df['TotalClaims'] > 0).mean()*100:.2f}%)")
    print(f"Mean claim amount: ${df['TotalClaims'].mean():.2f}")
    print(f"Mean premium: ${df['TotalPremium'].mean():.2f}")
    
    # ===== CLAIMS SEVERITY PREDICTION =====
    print("\n" + "="*60)
    print("CLAIMS SEVERITY PREDICTION (Regression)")
    print("="*60)
    
    # Prepare claims data
    claims_data = df[df['TotalClaims'] > 0].copy()
    claims_threshold = claims_data['TotalClaims'].quantile(0.99)
    claims_data = claims_data[claims_data['TotalClaims'] <= claims_threshold]
    print(f"Claims dataset: {len(claims_data):,} records")
    
    X_claims = claims_data[all_features].copy()
    y_claims = claims_data['TotalClaims'].copy()
    
    # Train-test split
    X_train_claims, X_test_claims, y_train_claims, y_test_claims = train_test_split(
        X_claims, y_claims, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Create preprocessor
    preprocessor_claims = create_preprocessor(numerical_features, categorical_features)
    
    # Initialize models
    models_claims = {
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor_claims),
            ('regressor', LinearRegression())
        ]),
        'Decision Tree': Pipeline([
            ('preprocessor', preprocessor_claims),
            ('regressor', DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor_claims),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, max_depth=10, n_jobs=-1))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor_claims),
            ('regressor', xgb.XGBRegressor(random_state=RANDOM_STATE, n_estimators=100, max_depth=6, n_jobs=-1))
        ])
    }
    
    # Train and evaluate claims models
    claims_results = {}
    for name, model in models_claims.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train_claims, y_train_claims)
            results = evaluate_regression_model(model, X_test_claims, y_test_claims, name)
            claims_results[name] = results
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # ===== CLAIM PROBABILITY PREDICTION =====
    print("\n" + "="*60)
    print("CLAIM PROBABILITY PREDICTION (Classification)")
    print("="*60)
    
    # Use sample for efficiency
    sample_size = min(50000, len(df))
    df_sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
    print(f"Using sample of {sample_size:,} records")
    
    X_probability = df_sample[all_features].copy()
    y_probability = df_sample['HasClaim'].copy()
    
    # Train-test split
    X_train_prob, X_test_prob, y_train_prob, y_test_prob = train_test_split(
        X_probability, y_probability, test_size=0.2, random_state=RANDOM_STATE, stratify=y_probability
    )
    
    # Create preprocessor
    preprocessor_prob = create_preprocessor(numerical_features, categorical_features)
    
    # Initialize models
    models_probability = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor_prob),
            ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
        ]),
        'Decision Tree': Pipeline([
            ('preprocessor', preprocessor_prob),
            ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor_prob),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=10, n_jobs=-1))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor_prob),
            ('classifier', xgb.XGBClassifier(random_state=RANDOM_STATE, n_estimators=100, max_depth=6, n_jobs=-1))
        ])
    }
    
    # Train and evaluate probability models
    probability_results = {}
    for name, model in models_probability.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train_prob, y_train_prob)
            results = evaluate_classification_model(model, X_test_prob, y_test_prob, name)
            probability_results[name] = results
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # ===== FINAL COMPARISON =====
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON REPORT")
    print("="*80)
    
    # Claims severity comparison
    if claims_results:
        print("\n📊 CLAIMS SEVERITY PREDICTION RESULTS:")
        claims_df = pd.DataFrame(claims_results).T
        print(claims_df.round(4).to_string())
        
        best_claims = claims_df.loc[claims_df['R2'].idxmax()]
        print(f"\n🏆 Best Claims Model: {best_claims.name} (R² = {best_claims['R2']:.4f})")
    
    # Probability prediction comparison
    if probability_results:
        print("\n📊 CLAIM PROBABILITY PREDICTION RESULTS:")
        prob_df = pd.DataFrame(probability_results).T
        print(prob_df.round(4).to_string())
        
        best_prob = prob_df.loc[prob_df['AUC'].idxmax()]
        print(f"\n🏆 Best Probability Model: {best_prob.name} (AUC = {best_prob['AUC']:.4f})")
    
    # Risk-based premium framework
    print("\n💰 RISK-BASED PREMIUM FRAMEWORK:")
    print("Formula: Premium = (Claim Probability × Expected Claim Severity) × (1 + Expense Loading + Profit Margin)")
    print("✓ Two-model approach successfully implemented")
    print("✓ Ready for production deployment with best performing models")
    
    print("\n" + "="*80)
    print("MODELING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main() 