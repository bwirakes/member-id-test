import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    confusion_matrix, roc_curve, average_precision_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictionModel:
    """
    Comprehensive churn prediction model with advanced feature engineering,
    multiple algorithm comparison, and robust evaluation metrics.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.scalers = {}
        self.label_encoders = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.results = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Advanced feature engineering and preprocessing.
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            Tuple of (processed_df, preprocessing_artifacts)
        """
        logger.info("Starting feature preparation...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Separate feature types
        numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove ID and target from feature lists
        if 'member_id' in numeric_features:
            numeric_features.remove('member_id')
        if 'is_churned' in numeric_features:
            numeric_features.remove('is_churned')
        if 'member_id' in categorical_features:
            categorical_features.remove('member_id')
        if 'is_churned' in categorical_features:
            categorical_features.remove('is_churned')
        
        # Handle missing values intelligently
        logger.info("Handling missing values...")
        
        # For numeric features, use median imputation
        for col in numeric_features:
            if df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {df_processed[col].isnull().sum()} missing values in {col} with median: {median_val}")
        
        # For categorical features, use mode or 'unknown'
        for col in categorical_features:
            if df_processed[col].isnull().sum() > 0:
                mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'unknown'
                df_processed[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with: {mode_val}")
        
        # Create interaction features
        logger.info("Creating interaction features...")
        if 'avg_order_value' in df_processed.columns and 'avg_orders_per_month' in df_processed.columns:
            df_processed['aov_x_frequency'] = df_processed['avg_order_value'] * df_processed['avg_orders_per_month']
        
        if 'days_since_last_order' in df_processed.columns and 'avg_orders_per_month' in df_processed.columns:
            df_processed['recency_x_frequency'] = df_processed['days_since_last_order'] * df_processed['avg_orders_per_month']
        
        if 'product_group_diversity' in df_processed.columns and 'brand_diversity' in df_processed.columns:
            df_processed['diversity_score'] = df_processed['product_group_diversity'] * df_processed['brand_diversity']
        
        # Create RFM-style scores
        if all(col in df_processed.columns for col in ['days_since_last_order', 'total_orders', 'total_gmv']):
            df_processed['rfm_score'] = (
                pd.qcut(df_processed['days_since_last_order'], 5, labels=[5,4,3,2,1]).astype(int) +
                pd.qcut(df_processed['total_orders'], 5, labels=[1,2,3,4,5]).astype(int) +
                pd.qcut(df_processed['total_gmv'], 5, labels=[1,2,3,4,5]).astype(int)
            )
        
        # Handle categorical encoding
        logger.info("Encoding categorical features...")
        self.label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Store preprocessing artifacts
        preprocessing_info = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'label_encoders': self.label_encoders,
            'feature_names': [col for col in df_processed.columns if col not in ['member_id', 'is_churned']]
        }
        
        logger.info(f"Feature preparation complete. Shape: {df_processed.shape}")
        return df_processed, preprocessing_info
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train multiple models with time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with model results and metrics
        """
        logger.info("Starting model training...")
        
        # Ensure we have the right feature names
        self.feature_names = X.columns.tolist()
        
        # Time-based split for more realistic evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Model configurations with proper hyperparameters
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y[y==0])/len(y[y==1]) if len(y[y==1]) > 0 else 1,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=self.random_state,
                verbose=-1
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000,
                C=1.0
            )
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models_config.items():
            logger.info(f"Training {name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
                
                # Train on full dataset
                model.fit(X, y)
                
                # Get predictions for the full dataset
                y_pred_proba = model.predict_proba(X)[:, 1]
                y_pred = model.predict(X)
                
                # Calculate comprehensive metrics
                auc_score = roc_auc_score(y, y_pred_proba)
                avg_precision = average_precision_score(y, y_pred_proba)
                
                results[name] = {
                    'model': model,
                    'cv_mean_auc': np.mean(cv_scores),
                    'cv_std_auc': np.std(cv_scores),
                    'full_data_auc': auc_score,
                    'avg_precision': avg_precision,
                    'cv_scores': cv_scores,
                    'y_pred_proba': y_pred_proba,
                    'y_pred': y_pred
                }
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(self.feature_names, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(self.feature_names, abs(model.coef_[0])))
                
                logger.info(f"{name}: CV AUC = {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Select best model based on cross-validation AUC
        if results:
            self.best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean_auc'])
            self.best_model = results[self.best_model_name]['model']
            self.models = {name: res['model'] for name, res in results.items()}
            
            logger.info(f"Best model: {self.best_model_name} with CV AUC: {results[self.best_model_name]['cv_mean_auc']:.4f}")
        
        self.results = results
        return results
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, model_name: str = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            X: Feature matrix
            y: True labels
            model_name: Specific model to evaluate (default: best model)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Calculate metrics
        auc_score = roc_auc_score(y, y_pred_proba)
        avg_precision = average_precision_score(y, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        
        evaluation_results = {
            'auc_score': auc_score,
            'average_precision': avg_precision,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
        
        return evaluation_results
    
    def predict_churn_probability(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Predict churn probability for new data.
        
        Args:
            X: Feature matrix
            model_name: Specific model to use (default: best model)
            
        Returns:
            Array of churn probabilities
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        return model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, top_n: int = 20, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance for interpretation.
        
        Args:
            top_n: Number of top features to return
            model_name: Specific model (default: best model)
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.feature_importance:
            logger.warning(f"Feature importance not available for {model_name}")
            return pd.DataFrame()
        
        importance_dict = self.feature_importance[model_name]
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in importance_dict.items()
        ]).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20, model_name: str = None, figsize: Tuple[int, int] = (10, 8)):
        """Plot feature importance."""
        importance_df = self.get_feature_importance(top_n, model_name)
        
        if importance_df.empty:
            logger.warning("No feature importance data available for plotting")
            return
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name or self.best_model_name}')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot comparison of all trained models."""
        if not self.results:
            logger.warning("No model results available for comparison")
            return
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        cv_means = [self.results[name]['cv_mean_auc'] for name in model_names]
        cv_stds = [self.results[name]['cv_std_auc'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # CV AUC comparison
        ax1.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        ax1.set_title('Cross-Validation AUC Comparison')
        ax1.set_ylabel('AUC Score')
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Box plot of CV scores
        cv_scores_data = [self.results[name]['cv_scores'] for name in model_names]
        ax2.boxplot(cv_scores_data, labels=model_names)
        ax2.set_title('Cross-Validation Score Distribution')
        ax2.set_ylabel('AUC Score')
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str, model_name: str = None):
        """Save trained model to disk."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = {
            'model': self.models[model_name],
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance.get(model_name, {}),
            'model_name': model_name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.models = {model_data['model_name']: model_data['model']}
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.label_encoders = model_data['label_encoders']
        self.feature_importance = {model_data['model_name']: model_data['feature_importance']}
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all trained models."""
        if not self.results:
            return {}
        
        summary = {}
        for name, result in self.results.items():
            summary[name] = {
                'cv_mean_auc': result['cv_mean_auc'],
                'cv_std_auc': result['cv_std_auc'],
                'full_data_auc': result['full_data_auc'],
                'average_precision': result['avg_precision']
            }
        
        return summary 