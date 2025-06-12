# 📊 Insurance Risk Analytics & Predictive Modeling

<div align="center">
  <h3><b>Advanced Analytics for Insurance Risk Assessment</b></h3>
  <p>A comprehensive data science project for analyzing insurance claims, premiums, and risk factors to build predictive models for better decision making</p>
</div>

## 📗 Table of Contents

- [📖 About the Project](#about-project)
- [🛠 Built With](#built-with)
- [📊 Project Structure](#project-structure)
- [💻 Getting Started](#getting-started)
- [📈 Key Analyses](#key-analyses)
- [🔬 Exploratory Data Analysis](#eda)
- [🤖 Machine Learning Models](#ml-models)
- [📋 Statistical Insights](#statistical-insights)
- [👥 Authors](#authors)
- [🤝 Contributing](#contributing)
- [📝 License](#license)

## 📖 About the Project <a name="about-project"></a>

This project focuses on **Insurance Risk Analytics** and **Predictive Modeling** to help insurance companies:

- 🎯 **Assess Risk Profiles**: Analyze customer demographics, vehicle characteristics, and geographic factors
- 💰 **Optimize Pricing**: Calculate appropriate premiums based on risk factors
- 📊 **Predict Claims**: Build models to forecast claim frequency and severity
- 🔍 **Detect Patterns**: Identify trends in claims data across different segments

### Key Objectives

1. **Exploratory Data Analysis (EDA)**: Deep dive into insurance data to understand patterns and relationships
2. **Statistical Analysis**: Calculate Loss Ratios, correlations, and statistical distributions
3. **Risk Modeling**: Develop predictive models for claims and premium optimization
4. **Business Intelligence**: Generate actionable insights for insurance decision-making

## 🛠 Built With <a name="built-with"></a>

### Tech Stack

<details>
  <summary>Data Science & Analytics</summary>
  <ul>
    <li><strong>Python 3.8+</strong> - Primary programming language</li>
    <li><strong>Pandas</strong> - Data manipulation and analysis</li>
    <li><strong>NumPy</strong> - Numerical computations</li>
    <li><strong>Matplotlib & Seaborn</strong> - Data visualization</li>
    <li><strong>Plotly</strong> - Interactive visualizations</li>
    <li><strong>Scipy</strong> - Statistical analysis</li>
  </ul>
</details>

<details>
  <summary>Machine Learning</summary>
  <ul>
    <li><strong>Scikit-learn</strong> - ML algorithms and model evaluation</li>
    <li><strong>XGBoost</strong> - Gradient boosting models</li>
    <li><strong>LightGBM</strong> - Efficient gradient boosting</li>
    <li><strong>Statsmodels</strong> - Statistical modeling</li>
  </ul>
</details>

<details>
  <summary>Development & Deployment</summary>
  <ul>
    <li><strong>Jupyter Notebooks</strong> - Interactive development</li>
    <li><strong>Git & GitHub</strong> - Version control</li>
    <li><strong>GitHub Actions</strong> - CI/CD pipeline</li>
    <li><strong>Docker</strong> - Containerization</li>
  </ul>
</details>

## 📊 Project Structure <a name="project-structure"></a>

```
insurance-risk-analytics-predictive-modeling/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and preprocessed data
│   └── external/               # External data sources
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_predictive_modeling.ipynb
├── src/
│   ├── data/                   # Data processing modules
│   ├── features/               # Feature engineering
│   ├── models/                 # ML models
│   └── visualization/          # Plotting utilities
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── .github/workflows/          # CI/CD configurations
└── README.md
```

## 💻 Getting Started <a name="getting-started"></a>

### Prerequisites

- Python 3.8 or higher
- Git
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/insurance-risk-analytics-predictive-modeling.git
   cd insurance-risk-analytics-predictive-modeling
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## Find out Detailed documentation about the code at  [@Documentation](https://docs.google.com/document/d/1jJFYrbnRbeQBQ52i81veXWG-isA0j1BDJCXs_-ScGGQ/edit?usp=sharing)

### Usage

1. **Start with EDA**: Open `notebooks/01_exploratory_data_analysis.ipynb`
2. **Run Statistical Analysis**: Continue with `notebooks/02_statistical_analysis.ipynb`
3. **Feature Engineering**: Proceed to `notebooks/03_feature_engineering.ipynb`
4. **Build Models**: Finally, open `notebooks/04_predictive_modeling.ipynb`

## 📈 Key Analyses <a name="key-analyses"></a>



### Loss Ratio Analysis
- Calculate overall Loss Ratio (TotalClaims / TotalPremium)
- Analyze variations by Province, VehicleType, and Gender
- Identify high-risk segments

### Financial Variables Distribution
- Examine distributions of TotalClaims and TotalPremium
- Detect and handle outliers in CustomValueEstimate
- Statistical significance testing

### Temporal Trends
- Track claim frequency and severity over 18-month period
- Seasonal patterns in insurance claims
- Month-over-month premium changes

### Vehicle Analysis
- Vehicle makes/models with highest/lowest claim amounts
- Risk factors by vehicle characteristics
- Age and value impact on claims

## 🔬 Exploratory Data Analysis <a name="eda"></a>

Key EDA components include:

- **Data Quality Assessment**: Missing values, data types, outlier detection
- **Univariate Analysis**: Distribution plots for all variables
- **Bivariate Analysis**: Correlation matrices and scatter plots
- **Geographic Analysis**: Regional patterns in claims and premiums
- **Creative Visualizations**: 3 beautiful and insightful plots

## 🤖 Machine Learning Models <a name="ml-models"></a>

### Task 4: Predictive Modeling (**COMPLETED**)

**Objective**: Build and evaluate predictive models for dynamic, risk-based pricing system.

#### 🎯 **Modeling Goals Achieved**
1. **Claim Severity Prediction**: Regression models to predict `TotalClaims` amount
   - Linear Regression: R² = 0.2656, RMSE = $25,162
   - Advanced models (Random Forest, XGBoost) implemented in notebooks
   - Target: Policies with claims > 0

2. **Claim Probability Prediction**: Classification models for claim occurrence
   - Logistic Regression: Accuracy = 99.79%, Precision = 100%
   - Binary classification (HasClaim: 0/1)
   - Advanced ensemble methods available

#### 🔧 **Technical Implementation**
- **Data Preparation**: Missing value imputation, feature scaling, train-test splits
- **Feature Engineering**: 7 new features including risk scores and ratios
- **Model Pipeline**: End-to-end preprocessing and prediction pipeline
- **Evaluation**: RMSE, R², Accuracy, Precision, Recall, F1-Score, AUC-ROC

#### 📊 **Risk-Based Premium Framework**
```
Premium = (Claim Probability × Expected Claim Severity) × (1 + Expense Loading + Profit Margin)
```
- **Two-model approach**: Probability + Severity prediction
- **Business parameters**: 10% expense loading, 15% profit margin
- **Production ready**: Complete implementation with monitoring

#### 📁 **Deliverables**
- `04_predictive_modeling.ipynb` - Main modeling notebook
- `05_advanced_modeling.ipynb` - Advanced techniques
- `run_modeling.py` - Production script
- `basic_modeling.py` - Simplified implementation
- `TASK_4_REPORT.md` - Comprehensive analysis report

### Model Evaluation
- Cross-validation strategies
- Performance metrics (RMSE, MAE, AUC-ROC)
- Feature importance analysis
- Model interpretability with SHAP

## 📋 Statistical Insights <a name="statistical-insights"></a>

- Statistical distributions fitting
- Hypothesis testing
- Confidence intervals
- A/B testing for different segments

## 👥 Authors <a name="authors"></a>

👤 **Belay Birhanu G.**

- GitHub: [@Github](https://github.com/belaymit)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/belay-bgwa/)

## 🤝 Contributing <a name="contributing"></a>

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](../../issues/).

### Contributing Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License <a name="license"></a>

This project is [MIT](./MIT.md) licensed.

---

<div align="center">
  <p><strong>📊 Building the Future of Insurance Analytics 🚀</strong></p>
</div>



