#!/usr/bin/env python3
"""
Example Usage Script for Restaurant Churn Prediction System

This script demonstrates various ways to use the churn prediction system,
from basic pipeline execution to advanced custom analysis.

Run this script after setting up your database configuration.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Local imports
from config import Config
from database import NeonDBConnector
from ml_model import ChurnPredictionModel
from churn_prediction_pipeline import ChurnPredictionPipeline
from data_analysis import ChurnDataAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_pipeline():
    """Example 1: Basic pipeline execution with default settings."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Pipeline Execution")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline()
    
    # Run complete pipeline
    success = pipeline.run_pipeline(save_outputs=True)
    
    if success:
        print("✅ Pipeline completed successfully!")
        print(f"📊 Best model: {pipeline.ml_model.best_model_name}")
        
        # Display basic results
        if 'churn_stats' in pipeline.results:
            stats = pipeline.results['churn_stats']
            print(f"📈 Churn rate: {stats['churn_rate']:.2%}")
            print(f"👥 Total customers: {stats['total_customers']:,}")
    else:
        print("❌ Pipeline failed!")

def example_2_custom_configuration():
    """Example 2: Custom configuration and model parameters."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    # Create custom config
    config = Config()
    config.CHURN_THRESHOLD = 0.3  # 30% reduction threshold
    config.CURRENT_PERIOD_DAYS = 90  # 90-day periods
    
    # Initialize pipeline with custom config
    pipeline = ChurnPredictionPipeline(config)
    
    print(f"🔧 Using custom churn threshold: {config.CHURN_THRESHOLD}")
    print(f"🔧 Using custom period: {config.CURRENT_PERIOD_DAYS} days")
    
    # Test database connection
    if pipeline.setup_database_connection():
        print("✅ Database connection successful!")
        
        # Explore data structure
        exploration = pipeline.explore_data()
        if exploration:
            print("📊 Data exploration completed")

def example_3_step_by_step_execution():
    """Example 3: Step-by-step pipeline execution with intermediate analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("=" * 60)
    
    pipeline = ChurnPredictionPipeline()
    
    try:
        # Step 1: Setup
        if not pipeline.setup_database_connection():
            print("❌ Database setup failed")
            return
        
        # Step 2: Extract churn labels
        churn_labels = pipeline.extract_churn_labels()
        print(f"📊 Extracted churn labels for {len(churn_labels):,} customers")
        
        # Analyze churn distribution
        churn_rate = churn_labels['is_churned'].mean()
        print(f"📈 Overall churn rate: {churn_rate:.2%}")
        
        # Step 3: Extract features
        features = pipeline.extract_features()
        print(f"🔍 Extracted {len(features.columns)} features")
        
        # Step 4: Create final dataset
        final_dataset = pipeline.create_final_dataset()
        print(f"🔄 Final dataset shape: {final_dataset.shape}")
        
        # Step 5: Train models
        training_results = pipeline.train_models()
        
        # Display model comparison
        print("\n📋 Model Performance Comparison:")
        for model_name, result in training_results.items():
            print(f"  {model_name}: AUC = {result['cv_mean_auc']:.4f} ± {result['cv_std_auc']:.4f}")
        
        # Step 6: Generate predictions
        predictions = pipeline.generate_predictions()
        print(f"🎯 Generated predictions for {len(predictions):,} customers")
        
    except Exception as e:
        logger.error(f"Error in step-by-step execution: {e}")

def example_4_data_analysis():
    """Example 4: Advanced data analysis and visualization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Advanced Data Analysis")
    print("=" * 60)
    
    # First, run a basic pipeline to get data
    pipeline = ChurnPredictionPipeline()
    
    if not pipeline.setup_database_connection():
        print("❌ Database setup failed")
        return
    
    try:
        # Get the data
        churn_labels = pipeline.extract_churn_labels()
        features = pipeline.extract_features()
        final_dataset = pipeline.create_final_dataset()
        
        # Initialize analyzer
        analyzer = ChurnDataAnalyzer(final_dataset)
        
        print("🔍 Performing advanced data analysis...")
        
        # Customer segmentation
        segments = analyzer.create_customer_segments(save_plots=False)
        if not segments.empty:
            print("\n📊 Customer Segments:")
            print(segments)
        
        # Generate insights
        insights = analyzer.generate_insights_report()
        
        print("\n💡 Key Insights:")
        if 'basic_stats' in insights:
            stats = insights['basic_stats']
            print(f"  • Total customers: {stats['total_customers']:,}")
            print(f"  • Churn rate: {stats['churn_rate']:.2%}")
            if stats['avg_gmv']:
                print(f"  • Average GMV: ${stats['avg_gmv']:.2f}")
        
        if 'risk_factors' in insights:
            risk = insights['risk_factors']
            print(f"  • Customers inactive >60 days: {risk['customers_inactive_60_days']:,}")
            print(f"  • Churn rate for inactive: {risk['churn_rate_inactive_customers']:.2%}")
        
    except Exception as e:
        logger.error(f"Error in data analysis: {e}")

def example_5_model_deployment():
    """Example 5: Model deployment and prediction on new data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Model Deployment")
    print("=" * 60)
    
    # Create sample new customer data (in real scenario, this would come from your database)
    print("🔮 Simulating prediction on new customer data...")
    
    # This would typically come from your feature engineering pipeline
    new_customer_data = pd.DataFrame({
        'days_since_last_order': [15, 45, 90, 120],
        'total_orders': [25, 10, 5, 2],
        'total_gmv': [2500.0, 800.0, 300.0, 100.0],
        'avg_order_value': [100.0, 80.0, 60.0, 50.0],
        'avg_orders_per_month': [3.0, 1.5, 0.8, 0.3],
        'product_group_diversity': [5, 3, 2, 1],
        'brand_diversity': [8, 4, 2, 1],
        'outlet_diversity': [2, 1, 1, 1]
    })
    
    print(f"📊 Sample data shape: {new_customer_data.shape}")
    print("📋 Sample customer profiles:")
    print(new_customer_data.head())
    
    # Note: In a real deployment, you would:
    # 1. Load a pre-trained model: model.load_model('path/to/saved/model.pkl')
    # 2. Apply the same feature engineering as used in training
    # 3. Generate predictions
    
    print("\n💡 For actual deployment:")
    print("  1. Save your trained model using: model.save_model('model.pkl')")
    print("  2. Load it later using: model.load_model('model.pkl')")
    print("  3. Apply same preprocessing to new data")
    print("  4. Generate predictions using: model.predict_churn_probability(X)")

def example_6_database_exploration():
    """Example 6: Database exploration and data quality checks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Database Exploration")
    print("=" * 60)
    
    try:
        # Initialize database connector
        db_connector = NeonDBConnector()
        
        if not db_connector.test_connection():
            print("❌ Database connection failed")
            return
        
        print("🔍 Exploring database structure...")
        
        # Check table structures
        for table in ['order_header', 'order_item']:
            try:
                table_info = db_connector.get_table_info(table)
                row_count = db_connector.get_table_row_count(table)
                
                print(f"\n📋 {table.upper()} Table:")
                print(f"  • Columns: {len(table_info)}")
                print(f"  • Rows: {row_count:,}")
                
                # Show column details
                print("  • Structure:")
                for _, row in table_info.head().iterrows():
                    print(f"    - {row['column_name']}: {row['data_type']}")
                
                # Date range for order_header
                if table == 'order_header':
                    date_info = db_connector.get_date_range(table, 'order_date')
                    print(f"  • Date range: {date_info['min_date']} to {date_info['max_date']}")
                    print(f"  • Unique dates: {date_info['unique_dates']:,}")
                
            except Exception as e:
                print(f"❌ Error exploring {table}: {e}")
        
        # Sample data quality query
        quality_query = """
        SELECT 
            COUNT(*) as total_orders,
            COUNT(DISTINCT member_id) as unique_customers,
            COUNT(*) FILTER (WHERE grand_total IS NULL) as null_amounts,
            COUNT(*) FILTER (WHERE member_id IS NULL) as null_member_ids,
            MIN(order_date) as earliest_date,
            MAX(order_date) as latest_date
        FROM order_header
        """
        
        quality_results = db_connector.execute_query(quality_query)
        print("\n📊 Data Quality Summary:")
        for col, val in quality_results.iloc[0].items():
            print(f"  • {col}: {val}")
            
    except Exception as e:
        logger.error(f"Error in database exploration: {e}")

def main():
    """Main function to run all examples."""
    print("🍽️  Restaurant Churn Prediction System - Examples")
    print("=" * 80)
    
    # Check configuration
    config = Config()
    if not config.validate_config():
        print("⚠️  Warning: Database configuration may be incomplete.")
        print("   Please update config.py or set environment variables before running examples.")
        print("   Required: NEON_DB_HOST, NEON_DB_NAME, NEON_DB_USER, NEON_DB_PASSWORD")
        print("\n   You can still run some examples to see the code structure.")
    
    # Run examples
    examples = [
        ("Basic Pipeline", example_1_basic_pipeline),
        ("Custom Configuration", example_2_custom_configuration),
        ("Step-by-Step Execution", example_3_step_by_step_execution),
        ("Data Analysis", example_4_data_analysis),
        ("Model Deployment", example_5_model_deployment),
        ("Database Exploration", example_6_database_exploration),
    ]
    
    print("\n📚 Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "=" * 80)
    
    # For demonstration, run a few examples
    # In practice, you would choose which examples to run
    
    try:
        # Always run configuration example (safe)
        example_2_custom_configuration()
        
        # Try database exploration (requires valid config)
        example_6_database_exploration()
        
        # Show deployment example (informational)
        example_5_model_deployment()
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Examples completed!")
    print("\n💡 Next steps:")
    print("  1. Update your database configuration in config.py")
    print("  2. Run: python churn_prediction_pipeline.py")
    print("  3. Analyze results using the ChurnDataAnalyzer class")
    print("  4. Deploy models in your production environment")

if __name__ == "__main__":
    main() 