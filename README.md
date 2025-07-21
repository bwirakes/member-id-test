# Restaurant Churn Prediction System

A comprehensive machine learning pipeline for predicting customer churn in restaurant loyalty programs using order and transaction data from Neon PostgreSQL database.

## 🎯 Project Overview

This system analyzes restaurant loyalty and spending data from `order_header` and `order_item` tables to predict customer churn with high accuracy. It implements advanced feature engineering, multiple ML algorithms, and provides actionable business insights.

## 🏗️ Architecture

```
├── config.py                      # Configuration and database settings
├── database.py                    # Neon PostgreSQL connector
├── ml_model.py                    # Machine learning models and training
├── churn_prediction_pipeline.py   # Main orchestration pipeline
├── data_analysis.py               # Data visualization and analysis
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## ✨ Features

### Data Engineering
- **Robust Database Connection**: Secure Neon PostgreSQL integration with connection pooling
- **Advanced Feature Engineering**: RFM analysis, behavioral patterns, product affinity
- **Automated Data Quality**: Missing value handling, outlier detection, data validation

### Machine Learning
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression
- **Time-Series Cross-Validation**: Realistic model evaluation for temporal data
- **Advanced Metrics**: AUC, Precision-Recall, Feature Importance, Risk Segmentation

### Business Intelligence
- **Customer Segmentation**: RFM-based customer categorization
- **Risk Assessment**: Low/Medium/High risk customer identification
- **Interactive Dashboards**: Plotly-based visualization for stakeholders
- **Actionable Insights**: Business-ready churn prevention recommendations

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd restaurant-churn-prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Configuration

Set up your Neon database credentials by setting environment variables:

```bash
export NEON_DB_HOST="your-neon-hostname.neon.tech"
export NEON_DB_NAME="your-database-name" 
export NEON_DB_USER="your-username"
export NEON_DB_PASSWORD="your-password"
export NEON_DB_PORT="5432"
export NEON_DB_SSLMODE="require"
```

Or update the credentials directly in `config.py`.

### 3. Run the Pipeline

```bash
# Execute the complete churn prediction pipeline
python churn_prediction_pipeline.py
```

This will:
- Connect to your Neon database
- Extract and engineer features
- Train multiple ML models
- Generate predictions and insights
- Save results and model artifacts

## 📊 Data Requirements

### Required Tables

**order_header**
```sql
- id (primary key)
- member_id (customer identifier)
- order_date (transaction date)
- grand_total (order value)
- outlet_name (store location)
- member_tier_when_transact (loyalty tier)
- order_number (unique order identifier)
```

**order_item**
```sql
- order_number (links to order_header)
- product_group (product category)
- brand_name (product brand)
- sku (product identifier)
- quantity (items purchased)
- price (unit price)
- paid_price (actual paid price)
```

### Data Quality Requirements
- Minimum 365 days of historical data
- At least 3 transactions per customer for churn labeling
- Non-null values for key fields (member_id, order_date, grand_total)

## 🧠 Machine Learning Approach

### Target Variable Definition
Churn is defined as customers with **50% or greater reduction** in spending using two methods:
1. **Recent vs Previous**: Last 60 days vs previous 60 days
2. **Recent vs Historical**: Last 60 days vs historical baseline

### Feature Categories

**Recency Features**
- Days since last order
- Customer age (days since first order)
- Recency buckets (very_recent, recent, moderate, old)

**Frequency Features**
- Total orders
- Average orders per month
- Active months
- Purchase intervals

**Monetary Features**
- Total GMV (Gross Merchandise Value)
- Average order value
- Spending trends (90-day comparisons)
- Order value variance

**Behavioral Features**
- Outlet diversity
- Product group diversity
- Brand diversity
- Day-of-week preferences
- Discount rate sensitivity

**Advanced Features**
- RFM scores and segments
- Interaction features (AOV × Frequency)
- Growth rates and trends
- Customer lifetime value indicators

### Model Selection
The system trains and compares 5 algorithms:
- **Random Forest**: Robust baseline with feature importance
- **XGBoost**: High-performance gradient boosting
- **LightGBM**: Fast, memory-efficient boosting
- **Gradient Boosting**: Traditional boosting approach
- **Logistic Regression**: Interpretable linear model

Best model selection based on cross-validated AUC score.

## 📈 Output and Results

### Generated Files
```
churn_predictions_YYYYMMDD_HHMMSS.csv    # Customer predictions
churn_model_YYYYMMDD_HHMMSS.pkl          # Trained model
churn_prediction_report_YYYYMMDD_HHMMSS.txt  # Performance report
churn_prediction_YYYYMMDD_HHMMSS.log     # Execution log
```

### Prediction Output Format
```csv
member_id,actual_churn,predicted_churn,churn_probability,risk_segment
12345,0,0,0.15,Low Risk
67890,1,1,0.85,High Risk
```

### Performance Metrics
- **AUC Score**: Area under ROC curve (target: >0.75)
- **Precision**: Accuracy of churn predictions
- **Recall**: Coverage of actual churners
- **F1 Score**: Harmonic mean of precision and recall

## 🔍 Usage Examples

### Basic Pipeline Execution
```python
from churn_prediction_pipeline import ChurnPredictionPipeline

# Initialize and run pipeline
pipeline = ChurnPredictionPipeline()
success = pipeline.run_pipeline(save_outputs=True)
```

### Custom Analysis
```python
from data_analysis import ChurnDataAnalyzer
import pandas as pd

# Load results and analyze
analyzer = ChurnDataAnalyzer()
analyzer.load_data('churn_predictions_20240101_120000.csv')

# Generate visualizations
analyzer.explore_churn_distribution()
analyzer.analyze_feature_correlations()
analyzer.create_customer_segments()

# Create interactive dashboard
analyzer.create_interactive_dashboard()
```

### Model Deployment
```python
from ml_model import ChurnPredictionModel

# Load trained model
model = ChurnPredictionModel()
model.load_model('churn_model_20240101_120000.pkl')

# Make predictions on new data
predictions = model.predict_churn_probability(new_customer_data)
```

## 🎯 Business Applications

### Retention Campaigns
- **High Risk Customers**: Immediate intervention with personalized offers
- **Medium Risk Customers**: Engagement campaigns and loyalty program perks
- **Low Risk Customers**: Maintain current experience, upsell opportunities

### Operational Insights
- **Product Strategy**: Identify products that drive loyalty
- **Outlet Performance**: Compare churn rates across locations
- **Tier Optimization**: Analyze effectiveness of loyalty tiers

### Revenue Impact
- **Proactive Retention**: Prevent high-value customer churn
- **Resource Allocation**: Focus retention efforts on recoverable customers
- **Lifetime Value**: Optimize customer acquisition costs

## 🔧 Advanced Configuration

### Model Hyperparameters
Customize model parameters in `ml_model.py`:

```python
models_config = {
    'xgboost': xgb.XGBClassifier(
        n_estimators=300,        # Increase for better performance
        learning_rate=0.05,      # Reduce for more conservative learning
        max_depth=8,             # Increase for more complex patterns
        # ... other parameters
    )
}
```

### Feature Engineering
Add custom features in the database queries or `ml_model.py`:

```python
# Example: Add seasonality features
df['month'] = pd.to_datetime(df['order_date']).dt.month
df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
```

### Churn Definition
Modify churn thresholds in `config.py`:

```python
class Config:
    CHURN_THRESHOLD = 0.3  # 30% reduction instead of 50%
    CURRENT_PERIOD_DAYS = 90  # 90-day periods instead of 60
```

## 🔍 Troubleshooting

### Common Issues

**Database Connection Failed**
```
Error: Database connection failed
Solution: Verify credentials in config.py or environment variables
```

**Insufficient Data**
```
Error: Not enough historical data
Solution: Ensure at least 365 days of data in order_header table
```

**Memory Issues**
```
Error: Memory allocation failed
Solution: Reduce dataset size or use chunking for large datasets
```

### Performance Optimization

**Large Datasets (>1M records)**
- Use data sampling for initial model development
- Implement batch processing for feature engineering
- Consider distributed computing frameworks

**Slow Model Training**
- Reduce cross-validation folds
- Use early stopping for tree-based models
- Parallelize with n_jobs=-1

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM recommended
- PostgreSQL database access

### Python Dependencies
See `requirements.txt` for complete list. Key packages:
- pandas, numpy (data processing)
- scikit-learn (machine learning)
- xgboost, lightgbm (advanced algorithms)
- psycopg2 (PostgreSQL connector)
- matplotlib, seaborn, plotly (visualization)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the execution logs for detailed error information

---

**Built with ❤️ for data-driven restaurant success** # member-id-test
