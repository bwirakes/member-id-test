import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    confusion_matrix, roc_curve, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple, List, Optional

import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictionModel:
    """
    Advanced hierarchical churn prediction model that trains specialized models for each
    customer tier, along with a global fallback model. Includes comprehensive feature
    engineering, validation, and business-focused optimization.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.tier_models = {}
        self.global_model = None
        self.tier_results = {}
        self.global_model_results = {}
        self.feature_names = {}  # Store feature names per tier
        self.label_encoders = {}
        self.scalers = {}  # Store scalers for numerical features if needed
        
        # Enhanced tier-specific weights based on business value
        self.tier_weights = {
            'VVIP': 15.0,    # Highest value customers
            'GOLD': 8.0,     # High value customers
            'SILVER': 4.0,   # Medium value customers
            'BLACK': 2.0,    # Standard customers
            'BRONZE': 1.5,   # Entry level customers
            'unknown': 1.0   # Default weight for other/unknown tiers
        }
        
        # Tier-specific minimum sample requirements
        self.min_samples_per_tier = {
            'VVIP': 50,      # Lower threshold for VIP tiers
            'GOLD': 75,
            'SILVER': 100,
            'BLACK': 150,
            'BRONZE': 150,
            'unknown': 100
        }

    def _validate_data(self, data: pd.DataFrame, tier: str = 'global') -> bool:
        """
        Validate input data for modeling.
        
        Args:
            data: DataFrame to validate
            tier: The customer tier being processed
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data.empty:
            logger.warning(f"Empty dataset provided for tier: {tier}")
            return False
            
        if 'member_tier_when_transact' not in data.columns and tier != 'global':
            logger.error(f"Missing tier column for hierarchical modeling in tier: {tier}")
            return False
            
        required_cols = ['member_id']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns for tier {tier}: {missing_cols}")
            return False
            
        return True

    def _engineer_tier_features(self, data: pd.DataFrame, tier: str) -> pd.DataFrame:
        """
        Engineers sophisticated features specific to a customer tier.
        
        Args:
            data: DataFrame for a specific tier.
            tier: The customer tier being processed.
            
        Returns:
            DataFrame with tier-specific features added.
        """
        if not self._validate_data(data, tier):
            return data.copy()
            
        base_features = data.copy()
        
        # VVIP-specific features: Focus on service quality and exclusivity
        if tier == 'VVIP':
            # Service engagement score
            if all(col in base_features.columns for col in ['outlet_diversity', 'product_group_diversity', 'brand_diversity']):
                base_features['vvip_engagement_score'] = (
                    base_features['outlet_diversity'] * 2 + 
                    base_features['product_group_diversity'] * 3 +
                    base_features['brand_diversity'] * 1.5
                )
            
            # Service expectation metric
            if all(col in base_features.columns for col in ['avg_order_value', 'avg_days_between_orders']):
                base_features['vvip_service_expectation'] = (
                    base_features['avg_order_value'] / (base_features['avg_days_between_orders'] + 1e-6)
                )
            
            # Premium behavior indicator
            if 'avg_discount_rate' in base_features.columns:
                base_features['vvip_premium_behavior'] = 1 / (base_features['avg_discount_rate'] + 0.01)

        # GOLD-specific features: Focus on loyalty and consistent spending
        elif tier == 'GOLD':
            # Loyalty consistency score
            if all(col in base_features.columns for col in ['active_months', 'avg_orders_per_month']):
                base_features['gold_loyalty_score'] = (
                    base_features['active_months'] * base_features['avg_orders_per_month']
                )
            
            # Spending stability
            if all(col in base_features.columns for col in ['order_value_std', 'avg_order_value']):
                base_features['gold_spending_stability'] = (
                    base_features['avg_order_value'] / (base_features['order_value_std'] + 1e-6)
                )

        # SILVER-specific features: Focus on growth potential and engagement
        elif tier == 'SILVER':
            # Growth potential score
            if all(col in base_features.columns for col in ['spend_growth_rate', 'brand_diversity']):
                base_features['silver_growth_potential'] = (
                    base_features['spend_growth_rate'] * base_features['brand_diversity']
                )
            
            # Engagement trend
            if all(col in base_features.columns for col in ['gmv_last_90_days', 'gmv_previous_90_days']):
                base_features['silver_engagement_trend'] = (
                    base_features['gmv_last_90_days'] / (base_features['gmv_previous_90_days'] + 1e-6)
                )

        # BLACK/BRONZE-specific features: Focus on price sensitivity and retention
        elif tier in ['BLACK', 'BRONZE']:
            # Price sensitivity score
            if 'avg_discount_rate' in base_features.columns:
                base_features['price_sensitivity'] = base_features['avg_discount_rate']
            
            # Value seeking behavior
            if all(col in base_features.columns for col in ['avg_order_value', 'avg_discount_rate']):
                base_features['value_seeking_score'] = (
                    base_features['avg_order_value'] * base_features['avg_discount_rate']
                )
                
        return base_features

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated interaction features."""
        df_processed = df.copy()
        
        # RFM interactions
        if all(col in df_processed.columns for col in ['days_since_last_order', 'total_orders', 'total_gmv']):
            # Recency-Frequency interaction
            df_processed['rf_interaction'] = (
                df_processed['total_orders'] / (df_processed['days_since_last_order'] + 1)
            )
            
            # Frequency-Monetary interaction
            df_processed['fm_interaction'] = df_processed['total_orders'] * df_processed['total_gmv']
            
            # RFM composite score
            df_processed['rfm_composite'] = (
                (1 / (df_processed['days_since_last_order'] + 1)) * 
                df_processed['total_orders'] * 
                df_processed['total_gmv']
            )

        # Behavioral consistency features
        if all(col in df_processed.columns for col in ['avg_days_between_orders', 'days_between_orders_std']):
            df_processed['purchase_regularity'] = (
                df_processed['avg_days_between_orders'] / (df_processed['days_between_orders_std'] + 1e-6)
            )

        # Channel loyalty
        if all(col in df_processed.columns for col in ['outlet_diversity', 'total_orders']):
            df_processed['channel_loyalty'] = (
                df_processed['total_orders'] / (df_processed['outlet_diversity'] + 1)
            )

        return df_processed

    def prepare_features(self, df: pd.DataFrame, tier: str = 'global') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Enhanced feature preparation with validation and comprehensive engineering.
        """
        logger.info(f"Starting feature preparation for tier: {tier}...")
        
        if not self._validate_data(df, tier):
            raise ValueError(f"Data validation failed for tier: {tier}")
        
        df_processed = df.copy()

        # Step 1: Tier-specific feature engineering
        if tier != 'global':
            df_processed = self._engineer_tier_features(df_processed, tier)

        # Step 2: Create interaction features
        df_processed = self._create_interaction_features(df_processed)

        # Step 3: Handle missing values with sophisticated imputation
        numeric_features = df_processed.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Clean up feature lists
        exclusion_cols = ['member_id', 'is_churned']
        numeric_features = [col for col in numeric_features if col not in exclusion_cols]
        categorical_features = [col for col in categorical_features if col not in exclusion_cols]

        # Enhanced numerical imputation
        for col in numeric_features:
            if df_processed[col].isnull().any():
                if col.endswith('_rate') or col.endswith('_ratio'):
                    # Use median for rates and ratios
                    fill_value = df_processed[col].median()
                elif col.startswith('days_'):
                    # Use forward fill then median for day-based features
                    df_processed[col] = df_processed[col].fillna(method='ffill').fillna(df_processed[col].median())
                    continue
                else:
                    # Use median for other numerical features
                    fill_value = df_processed[col].median()
                
                df_processed[col].fillna(fill_value, inplace=True)
        
        # Enhanced categorical imputation
        for col in categorical_features:
            if df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else 'unknown'
                df_processed[col].fillna(fill_value, inplace=True)
        
        # Step 4: Categorical Encoding with proper handling
        if tier not in self.label_encoders:
            self.label_encoders[tier] = {}
            
        for col in categorical_features:
            if col not in self.label_encoders[tier]:
                le = LabelEncoder()
                df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[tier][col] = le
            else:
                # Handle unseen categories for prediction
                le = self.label_encoders[tier][col]
                unique_values = set(df_processed[col].astype(str).unique())
                known_values = set(le.classes_)
                
                # Map unknown values to a default class
                if unknown_values := unique_values - known_values:
                    logger.warning(f"Unknown categories found in {col} for tier {tier}: {unknown_values}")
                    df_processed[col] = df_processed[col].astype(str).replace(
                        {val: le.classes_[0] for val in unknown_values}
                    )
                
                df_processed[col + '_encoded'] = le.transform(df_processed[col].astype(str))
        
        # Remove original categorical columns
        df_processed.drop(columns=categorical_features, inplace=True, errors='ignore')
        
        # Step 5: Feature selection and naming
        feature_names = [col for col in df_processed.columns if col not in exclusion_cols]
        
        preprocessing_info = {
            'feature_names': feature_names,
            'numeric_features': [f for f in feature_names if f in numeric_features or f.endswith('_encoded')],
            'tier': tier,
            'sample_size': len(df_processed)
        }
        
        logger.info(f"Feature preparation complete for {tier}. Shape: {df_processed.shape}, Features: {len(feature_names)}")
        return df_processed, preprocessing_info

    def train_hierarchical_models(self, df: pd.DataFrame):
        """
        Enhanced hierarchical model training with validation and optimization.
        """
        logger.info("Starting enhanced hierarchical model training...")
        
        # Validate input data
        if not self._validate_data(df):
            raise ValueError("Input data validation failed")
        
        tier_col = 'member_tier_when_transact'
        if tier_col not in df.columns:
            raise ValueError(f"Tier column '{tier_col}' not found in DataFrame.")
            
        df[tier_col] = df[tier_col].fillna('unknown')
        unique_tiers = df[tier_col].unique()
        
        logger.info(f"Found tiers: {list(unique_tiers)}")
        
        # Train tier-specific models
        successful_tiers = []
        for tier in unique_tiers:
            logger.info(f"--- Processing Tier: {tier} ---")
            tier_data = df[df[tier_col] == tier].copy()
            
            min_samples = self.min_samples_per_tier.get(tier, 100)
            if len(tier_data) < min_samples:
                logger.warning(f"Insufficient data for tier '{tier}' ({len(tier_data)} samples, need {min_samples}). Skipping dedicated model.")
                continue

            try:
                # Prepare features for this tier
                tier_data_processed, info = self.prepare_features(tier_data, tier=tier)
                X_tier = tier_data_processed[info['feature_names']]
                y_tier = tier_data_processed['is_churned']
                
                if y_tier.nunique() < 2:
                    logger.warning(f"Tier '{tier}' has no class variation. Skipping model training.")
                    continue
                
                self.feature_names[tier] = info['feature_names']

                # Calculate enhanced class weights
                churn_count = y_tier.sum()
                non_churn_count = len(y_tier) - churn_count
                
                if churn_count == 0:
                    logger.warning(f"No churned customers in tier '{tier}'. Skipping model.")
                    continue
                    
                churn_ratio = non_churn_count / churn_count if churn_count > 0 else 1
                business_weight = self.tier_weights.get(tier, 1.0)
                
                # Enhanced XGBoost model with tier-specific parameters
                if tier == 'VVIP':
                    # More conservative for VIP customers
                    model = xgb.XGBClassifier(
                        n_estimators=300, learning_rate=0.05, max_depth=4,
                        scale_pos_weight=churn_ratio * business_weight,
                        random_state=self.random_state, eval_metric='logloss', 
                        n_jobs=-1, subsample=0.8, colsample_bytree=0.8
                    )
                elif tier in ['GOLD', 'SILVER']:
                    # Balanced approach for mid-tier customers
                    model = xgb.XGBClassifier(
                        n_estimators=250, learning_rate=0.08, max_depth=5,
                        scale_pos_weight=churn_ratio * business_weight,
                        random_state=self.random_state, eval_metric='logloss', 
                        n_jobs=-1, subsample=0.85, colsample_bytree=0.85
                    )
                else:
                    # More aggressive for lower-tier customers
                    model = xgb.XGBClassifier(
                        n_estimators=200, learning_rate=0.1, max_depth=6,
                        scale_pos_weight=churn_ratio * business_weight,
                        random_state=self.random_state, eval_metric='logloss', 
                        n_jobs=-1, subsample=0.9, colsample_bytree=0.9
                    )
                
                # Train model
                logger.info(f"Training model for tier '{tier}' with {len(X_tier)} samples...")
                model.fit(X_tier, y_tier)
                self.tier_models[tier] = model
                successful_tiers.append(tier)
                
                # Evaluate model
                y_pred_proba = model.predict_proba(X_tier)[:, 1]
                auc = roc_auc_score(y_tier, y_pred_proba)
                
                self.tier_results[tier] = {
                    'auc': auc, 
                    'sample_size': len(X_tier), 
                    'churn_rate': y_tier.mean(),
                    'churn_count': churn_count,
                    'feature_count': len(info['feature_names'])
                }
                
                logger.info(f"Tier '{tier}' model trained successfully. AUC: {auc:.4f}, Churn Rate: {y_tier.mean():.2%}")
                
            except Exception as e:
                logger.error(f"Failed to train model for tier '{tier}': {e}")
                continue

        # Train global fallback model
        logger.info("--- Processing Global Fallback Model ---")
        try:
            df_global_processed, info = self.prepare_features(df, tier='global')
            X_global = df_global_processed[info['feature_names']]
            y_global = df_global_processed['is_churned']
            self.feature_names['global'] = info['feature_names']
            
            if y_global.nunique() < 2:
                raise ValueError("Global dataset has no class variation")
            
            churn_ratio = (len(y_global) - y_global.sum()) / y_global.sum() if y_global.sum() > 0 else 1
            
            logger.info(f"Training global fallback model with {len(X_global)} samples...")
            
            self.global_model = xgb.XGBClassifier(
                n_estimators=250, learning_rate=0.08, max_depth=6,
                scale_pos_weight=churn_ratio,
                random_state=self.random_state, eval_metric='logloss', 
                n_jobs=-1, subsample=0.85, colsample_bytree=0.85
            )
            self.global_model.fit(X_global, y_global)
            
            # Evaluate global model
            y_pred_proba_global = self.global_model.predict_proba(X_global)[:, 1]
            auc_global = roc_auc_score(y_global, y_pred_proba_global)
            
            self.global_model_results = {
                'auc': auc_global, 
                'sample_size': len(X_global), 
                'churn_rate': y_global.mean(),
                'churn_count': y_global.sum(),
                'feature_count': len(info['feature_names'])
            }
            
            logger.info(f"Global model trained successfully. AUC: {auc_global:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train global model: {e}")
            raise
        
        logger.info(f"Hierarchical training completed. Successfully trained models for {len(successful_tiers)} tiers + global model.")

    def predict_churn_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced prediction with better error handling and validation.
        """
        logger.info("Generating predictions with enhanced hierarchical models...")
        
        if df.empty:
            logger.warning("Empty dataframe provided for prediction")
            return pd.DataFrame(columns=['member_id', 'churn_probability'])
        
        if not self._validate_data(df):
            logger.error("Prediction data validation failed")
            return pd.DataFrame(columns=['member_id', 'churn_probability'])
            
        tier_col = 'member_tier_when_transact'
        df[tier_col] = df[tier_col].fillna('unknown')
        
        all_predictions = []
        prediction_stats = {}

        for tier, group in df.groupby(tier_col):
            logger.info(f"Predicting for tier: {tier} ({len(group)} samples)")
            
            # Choose model (tier-specific or global fallback)
            if tier in self.tier_models:
                model = self.tier_models[tier]
                tier_for_prepare = tier
                prediction_stats[tier] = 'tier_specific'
            else:
                model = self.global_model
                tier_for_prepare = 'global'
                prediction_stats[tier] = 'global_fallback'
                logger.info(f"Using global model for tier: {tier}")

            try:
                # Prepare features using the same preprocessing as training
                processed_group, info = self.prepare_features(group.copy(), tier=tier_for_prepare)
                
                # Align features with trained model
                model_features = self.feature_names[tier_for_prepare]
                
                # Handle missing features
                missing_features = set(model_features) - set(processed_group.columns)
                if missing_features:
                    logger.warning(f"Missing features for tier {tier}: {missing_features}")
                    for feature in missing_features:
                        processed_group[feature] = 0
                
                # Ensure correct column order and handle extra features
                processed_group = processed_group.reindex(columns=model_features, fill_value=0)
                
                # Generate predictions
                probabilities = model.predict_proba(processed_group[model_features])[:, 1]
                
                predictions_df = pd.DataFrame({
                    'member_id': group['member_id'],
                    'churn_probability': probabilities,
                    'prediction_tier': tier,
                    'model_used': prediction_stats[tier]
                })
                all_predictions.append(predictions_df)
                
            except Exception as e:
                logger.error(f"Prediction failed for tier {tier}: {e}")
                # Create default predictions
                predictions_df = pd.DataFrame({
                    'member_id': group['member_id'],
                    'churn_probability': 0.5,  # Default neutral probability
                    'prediction_tier': tier,
                    'model_used': 'default_fallback'
                })
                all_predictions.append(predictions_df)
        
        if not all_predictions:
            logger.error("No predictions generated")
            return pd.DataFrame(columns=['member_id', 'churn_probability'])

        # Combine all predictions
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Log prediction summary
        logger.info("Prediction Summary:")
        for tier, method in prediction_stats.items():
            count = len(final_predictions[final_predictions['prediction_tier'] == tier])
            logger.info(f"  {tier}: {count} predictions using {method}")
        
        return final_predictions[['member_id', 'churn_probability']]

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Enhanced feature importance analysis with validation.
        """
        all_importances = {}
        
        # Tier-specific model importance
        for tier, model in self.tier_models.items():
            try:
                if tier in self.feature_names and model is not None:
                    importance_dict = dict(zip(self.feature_names[tier], model.feature_importances_))
                    importance_df = pd.DataFrame.from_dict(
                        importance_dict, orient='index', columns=['importance']
                    ).sort_values('importance', ascending=False)
                    all_importances[tier] = importance_df.head(top_n)
            except Exception as e:
                logger.error(f"Failed to get feature importance for tier {tier}: {e}")
        
        # Global model importance
        if self.global_model and 'global' in self.feature_names:
            try:
                importance_dict = dict(zip(self.feature_names['global'], self.global_model.feature_importances_))
                importance_df = pd.DataFrame.from_dict(
                    importance_dict, orient='index', columns=['importance']
                ).sort_values('importance', ascending=False)
                all_importances['global'] = importance_df.head(top_n)
            except Exception as e:
                logger.error(f"Failed to get global model feature importance: {e}")
                
        return all_importances
    
    def save_model(self, filepath: str):
        """Enhanced model saving with validation."""
        try:
            model_data = {
                'tier_models': self.tier_models,
                'global_model': self.global_model,
                'feature_names': self.feature_names,
                'label_encoders': self.label_encoders,
                'tier_weights': self.tier_weights,
                'min_samples_per_tier': self.min_samples_per_tier,
                'tier_results': self.tier_results,
                'global_model_results': self.global_model_results
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Enhanced hierarchical model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Enhanced model loading with validation."""
        try:
            model_data = joblib.load(filepath)
            self.tier_models = model_data.get('tier_models', {})
            self.global_model = model_data.get('global_model')
            self.feature_names = model_data.get('feature_names', {})
            self.label_encoders = model_data.get('label_encoders', {})
            self.tier_weights = model_data.get('tier_weights', self.tier_weights)
            self.min_samples_per_tier = model_data.get('min_samples_per_tier', self.min_samples_per_tier)
            self.tier_results = model_data.get('tier_results', {})
            self.global_model_results = model_data.get('global_model_results', {})
            logger.info(f"Enhanced hierarchical model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
    def get_model_summary(self) -> Dict[str, Any]:
        """Enhanced model summary with comprehensive metrics."""
        summary = {
            'tiers': self.tier_results, 
            'global': self.global_model_results,
            'model_counts': {
                'tier_specific_models': len(self.tier_models),
                'global_fallback': 1 if self.global_model else 0,
                'total_features_global': len(self.feature_names.get('global', [])),
                'tier_feature_counts': {tier: len(features) for tier, features in self.feature_names.items() if tier != 'global'}
            }
        }
        return summary 