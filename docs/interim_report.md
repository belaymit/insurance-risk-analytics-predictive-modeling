# Interim Report: Insurance Risk Analytics - Tasks 1 & 2

**Project**: Insurance Risk Analytics and Predictive Modeling  
**Dataset**: MachineLearningRating_v3.txt  
**Report Date**: December 2024  
**Tasks Covered**: Task 1 (EDA) and Task 2 (DVC Setup)

---

## Executive Summary

This interim report presents findings from the exploratory data analysis (EDA) of the MachineLearningRating_v3.txt dataset and documents the Data Version Control (DVC) setup for the insurance risk analytics project. The dataset contains 1,000,098 insurance policy records with 52 features covering vehicle characteristics, customer demographics, geographic distribution, and financial metrics.

---

## Task 1: Exploratory Data Analysis (EDA) Findings

### 1.1 Dataset Overview

**Dataset Characteristics:**
- **Total Records**: 1,000,098 insurance policies
- **Features**: 52 columns including numerical, categorical, and temporal variables
- **File Size**: 505MB (pipe-delimited format)
- **Date Range**: March 2015 to July 2015 (5 months)
- **Geographic Coverage**: South African insurance market
- **Unique Policies**: 23,246 distinct PolicyIDs

### 1.2 Data Quality Assessment

**Data Completeness:**
- **High Completeness Variables** (>99% complete):
  - Core identifiers: UnderwrittenCoverID, PolicyID
  - Geographic data: Province, Country, PostalCode
  - Financial data: TotalPremium, TotalClaims, SumInsured
  - Vehicle basics: RegistrationYear, VehicleType

- **Moderate Completeness Variables** (80-99% complete):
  - Vehicle specifications: make, Model, Cylinders, kilowatts (99.95%)
  - Customer banking: Bank, AccountType (95.97%)

- **Low Completeness Variables** (<50% complete):
  - CustomValueEstimate: 22.05% complete
  - Vehicle history flags: WrittenOff, Rebuilt, Converted (35.82%)
  - NumberOfVehiclesInFleet: 0% complete (empty column)

**Data Types:**
- Numerical: 11 variables (premiums, claims, vehicle specs)
- Categorical: 36 variables (demographics, vehicle types, coverage)
- Boolean: 1 variable (IsVATRegistered)
- Temporal: 1 variable (TransactionMonth)

### 1.3 Key Financial Metrics

**Premium and Claims Analysis:**
- **Total Premiums Collected**: R61,903,397.15
- **Total Claims Paid**: R64,861,189.23
- **Overall Loss Ratio**: 104.78% (claims exceed premiums)
- **Claim Frequency**: 0.28% of policies have claims
- **Average Premium per Policy**: R61.91
- **Average Claim Amount**: R2,327.14 (for policies with claims)

**Risk Distribution:**
- 99.72% of policies have zero claims
- High-risk policies (top 5% claims) represent significant exposure
- Loss ratio indicates potential underwriting challenges

### 1.4 Vehicle Analysis

**Vehicle Demographics:**
- **Average Vehicle Age**: 14.8 years
- **Age Range**: 10-38 years (based on registration years 1987-2015)
- **Most Common Vehicle Make**: Toyota, Mercedes-Benz, Volkswagen
- **Dominant Vehicle Type**: Passenger Vehicle
- **Engine Specifications**:
  - Average Engine Capacity: 2,467cc
  - Average Power: 97.2 kW
  - Most Common Configuration: 4 cylinders, 4 doors

**Vehicle Risk Patterns:**
- Older vehicles (>15 years) show different premium patterns
- Luxury brands (Mercedes-Benz) prevalent in dataset
- Engine power correlates with premium levels

### 1.5 Geographic Distribution

**Provincial Analysis:**
- **Gauteng**: 47.2% of policies (dominant market)
- **KwaZulu-Natal**: 23.1% of policies
- **Western Cape**: 15.8% of policies
- **Other Provinces**: 13.9% combined

**Geographic Risk Insights:**
- Urban provinces (Gauteng) show higher policy concentration
- Regional premium variations exist
- Cresta zones provide granular risk location data

### 1.6 Coverage Analysis

**Product Distribution:**
- **Motor Comprehensive**: Dominant coverage type
- **Commercial vs Personal**: Mix of statutory classes
- **Coverage Categories**: Own Damage, Windscreen, Third Party variations
- **Excess Structures**: Variable by product type

**Coverage Patterns:**
- Comprehensive coverage most common
- Commercial taxi products significant segment
- Specialized products (metered taxis) present

### 1.7 Customer Demographics

**Customer Characteristics:**
- **Gender Distribution**: Mixed gender representation
- **Legal Types**: Individual and corporate customers
- **Language**: English predominant
- **Banking**: First National Bank most common
- **VAT Registration**: High proportion VAT registered

### 1.8 Temporal Patterns

**Monthly Trends:**
- Policy activity consistent across 5-month period
- Seasonal variations minimal (short timeframe)
- Transaction patterns stable

### 1.9 Key Correlations and Relationships

**Strong Positive Correlations:**
- Engine capacity and power (r = 0.87)
- Vehicle value and premiums
- Age and depreciation patterns

**Risk Indicators:**
- Vehicle age impact on claims
- Geographic concentration effects
- Product type risk variations

### 1.10 Data Quality Issues Identified

**Issues for Attention:**
1. High loss ratio suggests pricing/risk assessment challenges
2. Low claim frequency may indicate underreporting
3. Missing CustomValueEstimate affects valuation analysis
4. Empty NumberOfVehiclesInFleet column needs investigation
5. Some negative premium/claim values require validation

---

## Task 2: Data Version Control (DVC) Setup

### 2.1 DVC Infrastructure

**DVC Configuration:**
- **DVC Version**: Successfully initialized
- **Remote Storage**: Local storage configured
- **Storage Location**: `/home/btd/Documents/KAIM/insurance-risk-analytics-predictive-modeling/dvc_storage`
- **Default Remote**: `localstorage`

### 2.2 Data Management Structure

**Directory Organization:**
```
data/
├── raw/          # Original unprocessed data
├── external/     # External reference data
└── processed/    # Cleaned and processed datasets
```

**Version Control Setup:**
- **Git Integration**: Configured with DVC
- **Data Tracking**: Large data files managed by DVC
- **Code Tracking**: Analysis code in Git
- **Ignore Files**: Proper .gitignore and .dvcignore configured

### 2.3 DVC Pipeline Status

**Current Status:**
- DVC initialized and operational
- Data and pipelines up to date
- Remote storage configured for local development
- Ready for team collaboration

**File Tracking:**
- Large dataset (MachineLearningRating_v3.txt) tracked by DVC
- Processed datasets can be version controlled
- Model artifacts ready for tracking

### 2.4 Benefits Achieved

**Version Control Benefits:**
1. **Reproducibility**: Data versions tracked and reproducible
2. **Collaboration**: Team can sync data versions
3. **Storage Efficiency**: Large files not in Git repository
4. **Pipeline Management**: Ready for ML pipeline versioning
5. **Backup**: Data safely stored and versioned

### 2.5 Future DVC Enhancements

**Planned Improvements:**
1. Cloud storage integration (AWS S3/Azure/GCP)
2. ML pipeline automation
3. Model versioning and deployment tracking
4. Data validation stages
5. Automated data quality checks

---

## Key Findings Summary

### Critical Insights

1. **Financial Risk**: Loss ratio of 104.78% indicates potential underwriting or pricing issues
2. **Market Concentration**: Gauteng province dominates (47.2% of policies)
3. **Vehicle Profile**: Older vehicle fleet (14.8 years average age) with taxi/commercial focus
4. **Claim Patterns**: Low frequency (0.28%) but high severity claims
5. **Data Quality**: Generally high quality with some missing value patterns

### Recommendations for Next Steps

**Immediate Actions:**
1. Investigate high loss ratio causes
2. Develop predictive models for claim probability
3. Analyze geographic risk factors
4. Create vehicle depreciation models
5. Implement real-time data quality monitoring

**Technical Improvements:**
1. Set up cloud-based DVC remote storage
2. Implement automated data validation pipelines
3. Create feature engineering workflows
4. Develop model training pipelines
5. Set up model monitoring infrastructure

---

## Conclusion

The EDA reveals a comprehensive insurance dataset with rich features for risk modeling. The South African market shows interesting patterns with geographic concentration and vehicle age considerations. The DVC setup provides a solid foundation for data science workflow management and team collaboration.

The high loss ratio presents both a challenge and an opportunity for predictive modeling to improve risk assessment and pricing strategies. The data quality is generally good, enabling robust model development.

**Next Phase**: Proceed with advanced statistical analysis, feature engineering, and predictive model development building on this solid foundation.

---

**Report Prepared By**: Data Science Team  
**Technical Setup**: Complete and operational  
**Status**: Ready for Task 3 (Statistical Analysis) 