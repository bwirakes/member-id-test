#!/usr/bin/env python3
"""
Churn Prediction Pipeline for Restaurant Loyalty Data

This script orchestrates the complete machine learning pipeline for predicting customer churn
based on loyalty and spending data from order_header and order_item tables.

Usage:
    python churn_prediction_pipeline.py
    
Environment Variables:
    Set the following environment variables or update config.py:
    - NEON_DB_HOST
    - NEON_DB_NAME  
    - NEON_DB_USER
    - NEON_DB_PASSWORD
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings

# Local imports
from config import Config
from database import NeonDBConnector
from ml_model import ChurnPredictionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'churn_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class ChurnPredictionPipeline:
    """
    Complete pipeline for churn prediction including data extraction,
    feature engineering, model training, and evaluation.
    """
    
    def __init__(self, config: Config = None):
        """Initialize the pipeline with configuration."""
        self.config = config or Config()
        self.db_connector = None
        self.ml_model = None
        self.churn_labels = None
        self.features = None
        self.tier_summary = None
        self.cohort_analysis = None
        self.final_dataset = None
        self.results = {}
        
    def setup_database_connection(self) -> bool:
        """Setup and test database connection."""
        logger.info("Setting up database connection...")
        
        try:
            # Validate configuration
            if not self.config.validate_config():
                logger.error("Invalid database configuration. Please update config.py or set environment variables.")
                logger.error("Required: NEON_DB_HOST, NEON_DB_NAME, NEON_DB_USER, NEON_DB_PASSWORD")
                return False
            
            # Initialize database connector
            self.db_connector = NeonDBConnector(self.config)
            
            # Test connection
            if self.db_connector.test_connection():
                logger.info("Database connection successful!")
                return True
            else:
                logger.error("Database connection failed!")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up database connection: {e}")
            return False
    
    def explore_data(self) -> Dict[str, Any]:
        """Explore the database to understand data structure and quality."""
        logger.info("Starting data exploration...")
        
        exploration_results = {}
        
        try:
            # Get table information
            for table in ['order_header', 'order_item']:
                logger.info(f"Exploring {table} table...")
                table_info = self.db_connector.get_table_info(table)
                exploration_results[f'{table}_structure'] = table_info
                row_count = self.db_connector.get_table_row_count(table)
                exploration_results[f'{table}_row_count'] = row_count
                logger.info(f"{table} has {row_count:,} rows")
                
                if table == 'order_header':
                    date_info = self.db_connector.get_date_range(table, 'order_date')
                    exploration_results[f'{table}_date_range'] = date_info
                    if date_info:
                        logger.info(f"Date range: {date_info.get('min_date')} to {date_info.get('max_date')}")

            # This query was causing a type casting error. Correcting it.
            sample_query = """
            SELECT 
                COUNT(DISTINCT member_id) as unique_customers,
                COUNT(*) as total_orders,
                AVG(grand_total) as avg_order_value,
                MIN(order_date::timestamp::date) as first_order,
                MAX(order_date::timestamp::date) as last_order
            FROM order_header
            WHERE order_date::timestamp::date >= CURRENT_DATE - INTERVAL '365 days'
            """
            
            summary_stats = self.db_connector.execute_query(sample_query)
            if not summary_stats.empty:
                exploration_results['summary_stats'] = summary_stats.iloc[0].to_dict()
            
            logger.info("Data exploration completed successfully")
            
        except Exception as e:
            logger.error(f"Error during data exploration: {e}")
            
        return exploration_results
    
    def extract_churn_labels(self) -> pd.DataFrame:
        """Extract churn labels using the SQL logic."""
        logger.info("Extracting churn labels...")
        
        try:
            self.churn_labels = self.db_connector.create_churn_labels()
            
            # Log churn statistics
            churn_rate = self.churn_labels['is_churned'].mean()
            total_customers = len(self.churn_labels)
            churned_customers = self.churn_labels['is_churned'].sum()
            
            logger.info(f"Churn labels extracted successfully:")
            logger.info(f"  - Total customers: {total_customers:,}")
            logger.info(f"  - Churned customers: {churned_customers:,}")
            logger.info(f"  - Churn rate: {churn_rate:.2%}")
            
            # Store results
            self.results['churn_stats'] = {
                'total_customers': total_customers,
                'churned_customers': churned_customers,
                'churn_rate': churn_rate
            }
            
            return self.churn_labels
            
        except Exception as e:
            logger.error(f"Error extracting churn labels: {e}")
            raise
    
    def extract_features(self) -> pd.DataFrame:
        """Extract features using the SQL logic."""
        logger.info("Extracting features...")
        
        try:
            self.features = self.db_connector.create_feature_set()
            
            logger.info(f"Features extracted successfully:")
            logger.info(f"  - Number of customers: {len(self.features):,}")
            logger.info(f"  - Number of features: {len(self.features.columns) - 1}")  # -1 for member_id
            
            return self.features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def extract_cohort_analysis(self) -> pd.DataFrame:
        """Extracts detailed cohort analysis data."""
        logger.info("Extracting cohort analysis data...")
        try:
            self.cohort_analysis = self.db_connector.get_cohort_analysis()
            logger.info("Cohort analysis extracted successfully.")
            return self.cohort_analysis
        except Exception as e:
            logger.error(f"Error extracting cohort analysis: {e}")
            raise
    
    def create_final_dataset(self) -> pd.DataFrame:
        """Combine churn labels and features into final modeling dataset."""
        logger.info("Creating final dataset...")
        
        try:
            # Ensure features includes the tier column for hierarchical modeling
            if 'member_tier_when_transact' not in self.features.columns:
                 raise ValueError("Feature set must include 'member_tier_when_transact' for hierarchical modeling.")

            self.final_dataset = pd.merge(
                self.churn_labels[['member_id', 'is_churned']],
                self.features,
                on='member_id',
                how='inner'
            )
            
            logger.info(f"Final dataset created:")
            logger.info(f"  - Total samples: {len(self.final_dataset):,}")
            logger.info(f"  - Features: {len(self.final_dataset.columns) - 2}")
            logger.info(f"  - Missing values: {self.final_dataset.isnull().sum().sum()}")
            logger.info(f"  - Tiers found: {self.final_dataset['member_tier_when_transact'].unique()}")
            
            return self.final_dataset
            
        except Exception as e:
            logger.error(f"Error creating final dataset: {e}")
            raise
    
    def train_models(self) -> Dict[str, Any]:
        """Train hierarchical models: one for each tier and a global fallback."""
        logger.info("Starting hierarchical model training...")
        
        try:
            self.ml_model = ChurnPredictionModel(random_state=self.config.RANDOM_STATE)
            
            # The new method handles all training logic internally
            self.ml_model.train_hierarchical_models(self.final_dataset)
            
            training_results = self.ml_model.get_model_summary()
            self.results['training_results'] = training_results
            
            # Get feature importance for all models
            self.results['feature_importance'] = self.ml_model.get_feature_importance(top_n=20)
            
            logger.info("Hierarchical model training completed successfully!")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def generate_predictions(self, save_predictions: bool = True) -> pd.DataFrame:
        """Generate predictions for all customers using the hierarchical model."""
        logger.info("Generating predictions...")
        
        try:
            # The new prediction method takes the full dataframe and routes internally
            predictions_df = self.ml_model.predict_churn_probability(self.final_dataset)

            # Merge with actuals and create final prediction columns
            final_predictions = pd.merge(
                self.final_dataset[['member_id', 'is_churned']],
                predictions_df,
                on='member_id'
            ).rename(columns={'is_churned': 'actual_churn'})
            
            final_predictions['predicted_churn'] = (final_predictions['churn_probability'] >= 0.5).astype(int)
            final_predictions['risk_segment'] = pd.cut(
                final_predictions['churn_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                include_lowest=True
            )
            
            if save_predictions:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'churn_predictions_{timestamp}.csv'
                final_predictions.to_csv(filename, index=False)
                logger.info(f"Predictions saved to {filename}")
            
            self.results['predictions'] = final_predictions
            logger.info(f"Predictions generated for {len(final_predictions):,} customers")
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise
    
    def save_model(self, filepath: str = None) -> str:
        """Save the trained model."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'churn_model_{timestamp}.pkl'
        
        self.ml_model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report comparing hierarchical models."""
        logger.info("Generating final hierarchical report...")
        
        try:
            summary = self.ml_model.get_model_summary()
            report = {
                'pipeline_summary': {
                    'execution_time': datetime.now().isoformat(),
                    'data_shape': self.final_dataset.shape if self.final_dataset is not None else "N/A",
                    'model_type': 'Hierarchical XGBoost'
                },
                'cohort_analysis': self.cohort_analysis.to_dict('records') if self.cohort_analysis is not None else [],
                'data_summary': self.results.get('churn_stats', {}),
                'model_performance': summary,
                'top_features': self.results.get('feature_importance', {})
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f'churn_prediction_report_{timestamp}.txt'
            
            with open(report_filename, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("HIERARCHICAL CHURN PREDICTION PIPELINE REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Execution Time: {report['pipeline_summary']['execution_time']}\n")
                f.write(f"Dataset Shape: {report['pipeline_summary']['data_shape']}\n\n")
                
                f.write("COHORT ANALYSIS & CHURN RATES\n")
                f.write("-" * 80 + "\n")
                if report.get('cohort_analysis'):
                    for cohort_stats in report['cohort_analysis']:
                        tier = cohort_stats.get('tier', 'N/A').upper()
                        f.write(f"--- Cohort: {tier} ---\n")
                        f.write("  Visit-Based Churn:\n")
                        f.write(f"    - 120-Day Visit Churn Rate: {cohort_stats.get('churn_rate_120_day_visit', 0):.2%}\n")
                        f.write("  Spend-Based Churn (vs. Historical Avg):\n")
                        f.write(f"    - 30-Day Spend Churn Rate: {cohort_stats.get('churn_rate_30_day_spend', 0):.2%}\n")
                        f.write(f"    - 60-Day Spend Churn Rate: {cohort_stats.get('churn_rate_60_day_spend', 0):.2%}\n")
                        f.write(f"    - 90-Day Spend Churn Rate: {cohort_stats.get('churn_rate_90_day_spend', 0):.2%}\n")
                        f.write(f"    - 120-Day Spend Churn Rate: {cohort_stats.get('churn_rate_120_day_spend', 0):.2%}\n")
                        f.write("  Behavioral Metrics:\n")
                        f.write(f"    - Avg Monthly Spend: ${cohort_stats.get('avg_monthly_spend', 0):,.2f}\n")
                        f.write(f"    - Avg Monthly Visits: {cohort_stats.get('avg_monthly_visits', 0):,.2f}\n")
                        f.write(f"    - Avg Orders per Visit: {cohort_stats.get('avg_orders_per_visit', 0):,.2f}\n")
                        f.write(f"    - Avg Spend per Visit: ${cohort_stats.get('avg_spend_per_visit', 0):,.2f}\n\n")
                else:
                    f.write("  Cohort analysis not available.\n\n")

                f.write("MODEL PERFORMANCE SUMMARY\n")
                f.write("-" * 80 + "\n")
                
                # Global Model Performance
                global_perf = report['model_performance'].get('global', {})
                if global_perf:
                    f.write("--- Global Model (Fallback) ---\n")
                    f.write(f"  AUC on full dataset: {global_perf.get('auc', 'N/A'):.4f}\n")
                    f.write(f"  Sample Size: {global_perf.get('sample_size', 'N/A'):,}\n")
                    f.write(f"  Churn Rate in training data: {global_perf.get('churn_rate', 'N/A'):.2%}\n\n")

                # Tier-specific Model Performance
                tier_perf = report['model_performance'].get('tiers', {})
                if tier_perf:
                    f.write("--- Tier-Specific Models ---\n")
                    for tier, metrics in sorted(tier_perf.items()):
                        f.write(f"  Tier: {tier.upper()}\n")
                        f.write(f"    AUC on tier data: {metrics.get('auc', 'N/A'):.4f}\n")
                        f.write(f"    Sample Size: {metrics.get('sample_size', 'N/A'):,}\n")
                        f.write(f"    Churn Rate in tier data: {metrics.get('churn_rate', 'N/A'):.2%}\n\n")

                f.write("TOP 10 FEATURES BY MODEL\n")
                f.write("-" * 80 + "\n")
                if 'top_features' in report and report['top_features']:
                    for tier, importance_df in report['top_features'].items():
                        f.write(f"--- {tier.upper()} Model ---\n")
                        if not importance_df.empty:
                            for feature, importance in importance_df.head(10).iterrows():
                                f.write(f"  {feature:<30}: {importance['importance']:.4f}\n")
                        else:
                            f.write("  Importance not available.\n")
                        f.write("\n")

            logger.info(f"Report saved to {report_filename}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {}
    
    def load_existing_results(self, model_path: str, predictions_path: str):
        """Load existing model and predictions to regenerate a report."""
        logger.info(f"Loading existing model from {model_path}...")
        self.ml_model = ChurnPredictionModel()
        self.ml_model.load_model(model_path)
        
        logger.info(f"Loading existing predictions from {predictions_path}...")
        self.results['predictions'] = pd.read_csv(predictions_path)
        self.final_dataset = self.results['predictions'] # For shape reporting
        self.results['churn_stats'] = {
            'total_customers': len(self.final_dataset),
            'churned_customers': self.final_dataset['actual_churn'].sum(),
            'churn_rate': self.final_dataset['actual_churn'].mean()
        }

    def run_pipeline(self, save_outputs: bool = True) -> bool:
        """Run the complete churn prediction pipeline."""
        logger.info("Starting Churn Prediction Pipeline...")
        logger.info("=" * 80)
        
        try:
            if not self.setup_database_connection(): return False
            self.explore_data()
            self.extract_cohort_analysis()
            self.extract_churn_labels()
            self.extract_features()
            self.create_final_dataset()
            self.train_models()
            self.generate_predictions(save_predictions=save_outputs)
            if save_outputs: self.save_model()
            self.generate_report()
            
            logger.info("=" * 80)
            logger.info("Churn Prediction Pipeline completed successfully!")
            
            summary = self.ml_model.get_model_summary()
            if summary.get('global'):
                global_auc = summary['global']['auc']
                logger.info(f"Global Model AUC: {global_auc:.4f}")
            if summary.get('tiers'):
                logger.info("Tier-specific model AUCs:")
                for tier, metrics in summary['tiers'].items():
                    logger.info(f"  - {tier}: {metrics['auc']:.4f}")

            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            return False

def main():
    """Main function to run the churn prediction pipeline."""
    logger.info("Initializing Churn Prediction Pipeline...")
    
    pipeline = ChurnPredictionPipeline()
    success = pipeline.run_pipeline(save_outputs=True)
    
    if success:
        logger.info("Pipeline execution completed successfully!")
        # NOTE: Plotting functions were removed as they are not compatible
        # with the new hierarchical model structure without significant rework.
        # The detailed text report now provides model comparison insights.
    else:
        logger.error("Pipeline execution failed!")
        sys.exit(1)

def regenerate_report(model_path: str, predictions_path: str):
    """Function to regenerate a report from existing model and prediction files."""
    logger.info("Initializing report regeneration...")
    
    pipeline = ChurnPredictionPipeline()
    if not pipeline.setup_database_connection():
        logger.error("Failed to connect to the database. Aborting.")
        return

    pipeline.extract_cohort_analysis()
    pipeline.load_existing_results(model_path, predictions_path)
    pipeline.generate_report()
    
    logger.info("Report regeneration completed successfully!")

if __name__ == "__main__":
    main() 
    # Example of how to regenerate a report:
    # regenerate_report(
    #     model_path='churn_model_20250727_084311.pkl',
    #     predictions_path='churn_predictions_20250727_084311.csv'
    # ) 