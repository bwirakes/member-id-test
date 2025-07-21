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
                
                # Table structure
                table_info = self.db_connector.get_table_info(table)
                exploration_results[f'{table}_structure'] = table_info
                
                # Row count
                row_count = self.db_connector.get_table_row_count(table)
                exploration_results[f'{table}_row_count'] = row_count
                logger.info(f"{table} has {row_count:,} rows")
                
                # Date range for order_header
                if table == 'order_header':
                    date_info = self.db_connector.get_date_range(table, 'order_date')
                    exploration_results[f'{table}_date_range'] = date_info
                    logger.info(f"Date range: {date_info['min_date']} to {date_info['max_date']}")
            
            # Sample queries to understand data distribution
            sample_query = """
            SELECT 
                COUNT(DISTINCT member_id) as unique_customers,
                COUNT(*) as total_orders,
                AVG(grand_total) as avg_order_value,
                MIN(order_date) as first_order,
                MAX(order_date) as last_order
            FROM order_header
            WHERE order_date >= CURRENT_DATE - INTERVAL '365 days'
            """
            
            summary_stats = self.db_connector.execute_query(sample_query)
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
    
    def create_final_dataset(self) -> pd.DataFrame:
        """Combine churn labels and features into final modeling dataset."""
        logger.info("Creating final dataset...")
        
        try:
            # Merge churn labels and features
            self.final_dataset = pd.merge(
                self.churn_labels[['member_id', 'is_churned']],
                self.features,
                on='member_id',
                how='inner'
            )
            
            logger.info(f"Final dataset created:")
            logger.info(f"  - Total samples: {len(self.final_dataset):,}")
            logger.info(f"  - Features: {len(self.final_dataset.columns) - 2}")  # -2 for member_id and is_churned
            logger.info(f"  - Missing values: {self.final_dataset.isnull().sum().sum()}")
            
            # Check data balance
            churn_distribution = self.final_dataset['is_churned'].value_counts()
            logger.info(f"  - Class distribution: {dict(churn_distribution)}")
            
            return self.final_dataset
            
        except Exception as e:
            logger.error(f"Error creating final dataset: {e}")
            raise
    
    def train_models(self) -> Dict[str, Any]:
        """Train and evaluate multiple ML models."""
        logger.info("Starting model training...")
        
        try:
            # Initialize ML model
            self.ml_model = ChurnPredictionModel(random_state=self.config.RANDOM_STATE)
            
            # Prepare features
            features_df, preprocessing_info = self.ml_model.prepare_features(self.final_dataset)
            
            # Separate features and target
            X = features_df.drop(['member_id', 'is_churned'], axis=1, errors='ignore')
            y = features_df['is_churned']
            
            logger.info(f"Training dataset shape: {X.shape}")
            logger.info(f"Feature columns: {list(X.columns)}")
            
            # Train models
            training_results = self.ml_model.train_models(X, y)
            
            # Store results
            self.results['training_results'] = training_results
            self.results['preprocessing_info'] = preprocessing_info
            
            # Model evaluation
            evaluation_results = self.ml_model.evaluate_model(X, y)
            self.results['evaluation_results'] = evaluation_results
            
            # Feature importance
            feature_importance = self.ml_model.get_feature_importance(top_n=20)
            self.results['feature_importance'] = feature_importance
            
            logger.info("Model training completed successfully!")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def generate_predictions(self, save_predictions: bool = True) -> pd.DataFrame:
        """Generate predictions for all customers."""
        logger.info("Generating predictions...")
        
        try:
            # Prepare features for prediction
            features_df, _ = self.ml_model.prepare_features(self.final_dataset)
            X = features_df.drop(['member_id', 'is_churned'], axis=1, errors='ignore')
            
            # Generate predictions
            churn_probabilities = self.ml_model.predict_churn_probability(X)
            churn_predictions = (churn_probabilities >= 0.5).astype(int)
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'member_id': self.final_dataset['member_id'],
                'actual_churn': self.final_dataset['is_churned'],
                'predicted_churn': churn_predictions,
                'churn_probability': churn_probabilities
            })
            
            # Add risk segments
            predictions_df['risk_segment'] = pd.cut(
                predictions_df['churn_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            if save_predictions:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'churn_predictions_{timestamp}.csv'
                predictions_df.to_csv(filename, index=False)
                logger.info(f"Predictions saved to {filename}")
            
            self.results['predictions'] = predictions_df
            
            logger.info(f"Predictions generated for {len(predictions_df):,} customers")
            
            return predictions_df
            
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
        """Generate comprehensive report of the pipeline results."""
        logger.info("Generating final report...")
        
        try:
            report = {
                'pipeline_summary': {
                    'execution_time': datetime.now().isoformat(),
                    'data_shape': self.final_dataset.shape if self.final_dataset is not None else None,
                    'best_model': self.ml_model.best_model_name if self.ml_model else None
                },
                'data_summary': self.results.get('churn_stats', {}),
                'model_performance': self.ml_model.get_model_summary() if self.ml_model else {},
                'top_features': self.results.get('feature_importance', pd.DataFrame()).to_dict('records')
            }
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f'churn_prediction_report_{timestamp}.txt'
            
            with open(report_filename, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("CHURN PREDICTION PIPELINE REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Execution Time: {report['pipeline_summary']['execution_time']}\n")
                f.write(f"Dataset Shape: {report['pipeline_summary']['data_shape']}\n")
                f.write(f"Best Model: {report['pipeline_summary']['best_model']}\n\n")
                
                if 'churn_stats' in self.results:
                    stats = self.results['churn_stats']
                    f.write("DATA SUMMARY:\n")
                    f.write(f"  Total Customers: {stats['total_customers']:,}\n")
                    f.write(f"  Churned Customers: {stats['churned_customers']:,}\n")
                    f.write(f"  Churn Rate: {stats['churn_rate']:.2%}\n\n")
                
                if self.ml_model:
                    f.write("MODEL PERFORMANCE:\n")
                    model_summary = self.ml_model.get_model_summary()
                    for model_name, metrics in model_summary.items():
                        f.write(f"  {model_name}:\n")
                        f.write(f"    CV AUC: {metrics['cv_mean_auc']:.4f} ± {metrics['cv_std_auc']:.4f}\n")
                        f.write(f"    Full Data AUC: {metrics['full_data_auc']:.4f}\n")
                        f.write(f"    Average Precision: {metrics['average_precision']:.4f}\n\n")
                
                if not self.results.get('feature_importance', pd.DataFrame()).empty:
                    f.write("TOP FEATURES:\n")
                    for idx, row in self.results['feature_importance'].head(10).iterrows():
                        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
            
            logger.info(f"Report saved to {report_filename}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {}
    
    def run_pipeline(self, save_outputs: bool = True) -> bool:
        """Run the complete churn prediction pipeline."""
        logger.info("Starting Churn Prediction Pipeline...")
        logger.info("=" * 80)
        
        try:
            # Step 1: Setup database connection
            if not self.setup_database_connection():
                logger.error("Pipeline failed: Could not establish database connection")
                return False
            
            # Step 2: Explore data (optional)
            self.explore_data()
            
            # Step 3: Extract churn labels
            self.extract_churn_labels()
            
            # Step 4: Extract features
            self.extract_features()
            
            # Step 5: Create final dataset
            self.create_final_dataset()
            
            # Step 6: Train models
            self.train_models()
            
            # Step 7: Generate predictions
            self.generate_predictions(save_predictions=save_outputs)
            
            # Step 8: Save model
            if save_outputs:
                self.save_model()
            
            # Step 9: Generate report
            self.generate_report()
            
            logger.info("=" * 80)
            logger.info("Churn Prediction Pipeline completed successfully!")
            
            # Print summary
            if self.ml_model:
                best_model = self.ml_model.best_model_name
                best_auc = self.ml_model.results[best_model]['cv_mean_auc']
                logger.info(f"Best Model: {best_model} (CV AUC: {best_auc:.4f})")
            
            if 'churn_stats' in self.results:
                stats = self.results['churn_stats']
                logger.info(f"Churn Rate: {stats['churn_rate']:.2%} ({stats['churned_customers']:,}/{stats['total_customers']:,})")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            return False

def main():
    """Main function to run the churn prediction pipeline."""
    logger.info("Initializing Churn Prediction Pipeline...")
    
    # Create pipeline instance
    pipeline = ChurnPredictionPipeline()
    
    # Run the complete pipeline
    success = pipeline.run_pipeline(save_outputs=True)
    
    if success:
        logger.info("Pipeline execution completed successfully!")
        
        # Optional: Display feature importance plot
        if pipeline.ml_model:
            try:
                logger.info("Generating feature importance plot...")
                pipeline.ml_model.plot_feature_importance()
                
                logger.info("Generating model comparison plot...")
                pipeline.ml_model.plot_model_comparison()
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")
        
    else:
        logger.error("Pipeline execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 