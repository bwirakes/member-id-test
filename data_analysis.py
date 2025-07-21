import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any, List, Tuple, Optional

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnDataAnalyzer:
    """
    Comprehensive data analysis and visualization for churn prediction results.
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """Initialize with data."""
        self.data = data
        self.figures = {}
        
    def load_data(self, filepath: str = None, data: pd.DataFrame = None):
        """Load data from file or DataFrame."""
        if data is not None:
            self.data = data
        elif filepath:
            self.data = pd.read_csv(filepath)
        else:
            raise ValueError("Must provide either filepath or data DataFrame")
        
        logger.info(f"Data loaded with shape: {self.data.shape}")
    
    def explore_churn_distribution(self, figsize: Tuple[int, int] = (15, 10)):
        """Analyze churn distribution across various dimensions."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Churn Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Overall churn rate
        churn_counts = self.data['is_churned'].value_counts()
        axes[0, 0].pie(churn_counts.values, labels=['Not Churned', 'Churned'], autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Churn Distribution')
        
        # Churn by recency bucket
        if 'recency_bucket' in self.data.columns:
            churn_by_recency = self.data.groupby('recency_bucket')['is_churned'].agg(['count', 'mean'])
            axes[0, 1].bar(churn_by_recency.index, churn_by_recency['mean'])
            axes[0, 1].set_title('Churn Rate by Recency')
            axes[0, 1].set_ylabel('Churn Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Churn by tier
        if 'current_tier' in self.data.columns:
            churn_by_tier = self.data.groupby('current_tier')['is_churned'].agg(['count', 'mean'])
            axes[0, 2].bar(churn_by_tier.index, churn_by_tier['mean'])
            axes[0, 2].set_title('Churn Rate by Membership Tier')
            axes[0, 2].set_ylabel('Churn Rate')
        
        # RFM distribution
        if 'total_orders' in self.data.columns and 'total_gmv' in self.data.columns:
            churned = self.data[self.data['is_churned'] == 1]
            not_churned = self.data[self.data['is_churned'] == 0]
            
            axes[1, 0].scatter(not_churned['total_orders'], not_churned['total_gmv'], 
                             alpha=0.6, label='Not Churned', s=20)
            axes[1, 0].scatter(churned['total_orders'], churned['total_gmv'], 
                             alpha=0.6, label='Churned', s=20)
            axes[1, 0].set_xlabel('Total Orders')
            axes[1, 0].set_ylabel('Total GMV')
            axes[1, 0].set_title('RFM: Frequency vs Monetary')
            axes[1, 0].legend()
        
        # Days since last order distribution
        if 'days_since_last_order' in self.data.columns:
            churned = self.data[self.data['is_churned'] == 1]['days_since_last_order']
            not_churned = self.data[self.data['is_churned'] == 0]['days_since_last_order']
            
            axes[1, 1].hist(not_churned, bins=30, alpha=0.7, label='Not Churned', density=True)
            axes[1, 1].hist(churned, bins=30, alpha=0.7, label='Churned', density=True)
            axes[1, 1].set_xlabel('Days Since Last Order')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Recency Distribution')
            axes[1, 1].legend()
        
        # Average order value distribution
        if 'avg_order_value' in self.data.columns:
            churned = self.data[self.data['is_churned'] == 1]['avg_order_value']
            not_churned = self.data[self.data['is_churned'] == 0]['avg_order_value']
            
            # Use log scale for better visualization
            axes[1, 2].hist(np.log1p(not_churned), bins=30, alpha=0.7, label='Not Churned', density=True)
            axes[1, 2].hist(np.log1p(churned), bins=30, alpha=0.7, label='Churned', density=True)
            axes[1, 2].set_xlabel('Log(Average Order Value + 1)')
            axes[1, 2].set_ylabel('Density')
            axes[1, 2].set_title('AOV Distribution')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
        
        self.figures['churn_distribution'] = fig
    
    def analyze_feature_correlations(self, figsize: Tuple[int, int] = (12, 10)):
        """Analyze correlations between features and churn."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Select numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Show top correlations with churn
        if 'is_churned' in corr_matrix.columns:
            churn_corr = corr_matrix['is_churned'].abs().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_corr = churn_corr.head(15)
            sns.barplot(x=top_corr.values, y=top_corr.index, palette='viridis')
            plt.title('Top Features Correlated with Churn')
            plt.xlabel('Absolute Correlation with Churn')
            plt.tight_layout()
            plt.show()
    
    def create_customer_segments(self, save_plots: bool = True) -> pd.DataFrame:
        """Create customer segments based on RFM analysis."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        logger.info("Creating customer segments...")
        
        # Calculate RFM scores
        rfm_data = self.data.copy()
        
        # Recency (days since last order - lower is better)
        if 'days_since_last_order' in rfm_data.columns:
            rfm_data['R_score'] = pd.qcut(rfm_data['days_since_last_order'], 5, labels=[5,4,3,2,1])
        
        # Frequency (total orders - higher is better)
        if 'total_orders' in rfm_data.columns:
            rfm_data['F_score'] = pd.qcut(rfm_data['total_orders'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Monetary (total GMV - higher is better)
        if 'total_gmv' in rfm_data.columns:
            rfm_data['M_score'] = pd.qcut(rfm_data['total_gmv'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Create RFM segments
        if all(col in rfm_data.columns for col in ['R_score', 'F_score', 'M_score']):
            rfm_data['RFM_score'] = (
                rfm_data['R_score'].astype(int) +
                rfm_data['F_score'].astype(int) +
                rfm_data['M_score'].astype(int)
            )
            
            # Define segments
            def categorize_rfm(score):
                if score >= 13:
                    return 'Champions'
                elif score >= 11:
                    return 'Loyal Customers'
                elif score >= 9:
                    return 'Potential Loyalists'
                elif score >= 7:
                    return 'At Risk'
                elif score >= 5:
                    return 'Cannot Lose Them'
                else:
                    return 'Lost'
            
            rfm_data['segment'] = rfm_data['RFM_score'].apply(categorize_rfm)
            
            # Analyze churn by segment
            segment_analysis = rfm_data.groupby('segment').agg({
                'is_churned': ['count', 'mean'],
                'total_gmv': 'mean',
                'total_orders': 'mean',
                'days_since_last_order': 'mean'
            }).round(2)
            
            segment_analysis.columns = ['Customer_Count', 'Churn_Rate', 'Avg_GMV', 'Avg_Orders', 'Avg_Days_Since_Last']
            
            if save_plots:
                # Plot segment analysis
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Customer Segment Analysis', fontsize=16, fontweight='bold')
                
                # Segment distribution
                segment_counts = rfm_data['segment'].value_counts()
                axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
                axes[0, 0].set_title('Customer Segment Distribution')
                
                # Churn rate by segment
                axes[0, 1].bar(segment_analysis.index, segment_analysis['Churn_Rate'])
                axes[0, 1].set_title('Churn Rate by Segment')
                axes[0, 1].set_ylabel('Churn Rate')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Average GMV by segment
                axes[1, 0].bar(segment_analysis.index, segment_analysis['Avg_GMV'])
                axes[1, 0].set_title('Average GMV by Segment')
                axes[1, 0].set_ylabel('Average GMV')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Customer count by segment
                axes[1, 1].bar(segment_analysis.index, segment_analysis['Customer_Count'])
                axes[1, 1].set_title('Customer Count by Segment')
                axes[1, 1].set_ylabel('Customer Count')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.show()
            
            logger.info(f"Created {len(segment_analysis)} customer segments")
            return segment_analysis
        
        return pd.DataFrame()
    
    def plot_model_performance(self, predictions_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
        """Plot model performance metrics."""
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        
        if 'actual_churn' not in predictions_df.columns or 'churn_probability' not in predictions_df.columns:
            raise ValueError("Predictions DataFrame must contain 'actual_churn' and 'churn_probability' columns")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        y_true = predictions_df['actual_churn']
        y_prob = predictions_df['churn_probability']
        y_pred = predictions_df['predicted_churn'] if 'predicted_churn' in predictions_df.columns else (y_prob >= 0.5).astype(int)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_true, y_prob)
        
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(y_true, y_prob)
        
        axes[0, 1].plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
        axes[0, 2].set_title('Confusion Matrix')
        axes[0, 2].set_ylabel('Actual')
        axes[0, 2].set_xlabel('Predicted')
        
        # Probability Distribution
        churned = y_prob[y_true == 1]
        not_churned = y_prob[y_true == 0]
        
        axes[1, 0].hist(not_churned, bins=30, alpha=0.7, label='Not Churned', density=True)
        axes[1, 0].hist(churned, bins=30, alpha=0.7, label='Churned', density=True)
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('Churn Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Probability Distribution')
        axes[1, 0].legend()
        
        # Risk Segment Analysis
        if 'risk_segment' in predictions_df.columns:
            risk_analysis = predictions_df.groupby('risk_segment').agg({
                'actual_churn': ['count', 'mean'],
                'churn_probability': 'mean'
            }).round(3)
            
            risk_analysis.columns = ['Count', 'Actual_Churn_Rate', 'Avg_Probability']
            
            x_pos = range(len(risk_analysis))
            axes[1, 1].bar(x_pos, risk_analysis['Actual_Churn_Rate'], alpha=0.7, label='Actual Churn Rate')
            axes[1, 1].bar(x_pos, risk_analysis['Avg_Probability'], alpha=0.7, label='Avg Predicted Probability')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(risk_analysis.index, rotation=45)
            axes[1, 1].set_ylabel('Rate')
            axes[1, 1].set_title('Risk Segment Performance')
            axes[1, 1].legend()
        
        # Feature Importance (if available)
        axes[1, 2].text(0.5, 0.5, 'Feature Importance\n(Use plot_feature_importance\nfrom ML model)', 
                       ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        self.figures['model_performance'] = fig
    
    def create_interactive_dashboard(self, predictions_df: pd.DataFrame = None):
        """Create an interactive dashboard using Plotly."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Combine data with predictions if available
        dashboard_data = self.data.copy()
        if predictions_df is not None:
            dashboard_data = dashboard_data.merge(
                predictions_df[['member_id', 'churn_probability', 'risk_segment']], 
                on='member_id', 
                how='left'
            )
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Churn Rate by Segment', 'GMV Distribution', 
                           'Order Frequency vs Recency', 'Risk Segment Distribution',
                           'Feature Correlation with Churn', 'Customer Lifetime Value'],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Plot 1: Churn rate by segment
        if 'segment' in dashboard_data.columns:
            segment_churn = dashboard_data.groupby('segment')['is_churned'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=segment_churn['segment'], y=segment_churn['is_churned'],
                      name='Churn Rate', marker_color='lightcoral'),
                row=1, col=1
            )
        
        # Plot 2: GMV distribution
        if 'total_gmv' in dashboard_data.columns:
            fig.add_trace(
                go.Histogram(x=dashboard_data['total_gmv'], nbinsx=30, 
                           name='GMV Distribution', marker_color='lightblue'),
                row=1, col=2
            )
        
        # Plot 3: Order frequency vs recency
        if all(col in dashboard_data.columns for col in ['total_orders', 'days_since_last_order']):
            churned = dashboard_data[dashboard_data['is_churned'] == 1]
            not_churned = dashboard_data[dashboard_data['is_churned'] == 0]
            
            fig.add_trace(
                go.Scatter(x=not_churned['total_orders'], y=not_churned['days_since_last_order'],
                          mode='markers', name='Not Churned', marker=dict(color='green', size=4)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=churned['total_orders'], y=churned['days_since_last_order'],
                          mode='markers', name='Churned', marker=dict(color='red', size=4)),
                row=2, col=1
            )
        
        # Plot 4: Risk segment distribution
        if 'risk_segment' in dashboard_data.columns:
            risk_counts = dashboard_data['risk_segment'].value_counts()
            fig.add_trace(
                go.Pie(labels=risk_counts.index, values=risk_counts.values, 
                      name='Risk Segments'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Churn Prediction Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Show the dashboard
        fig.show()
        
        return fig
    
    def generate_insights_report(self, predictions_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate business insights from the analysis."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        insights = {}
        
        # Basic statistics
        insights['basic_stats'] = {
            'total_customers': len(self.data),
            'churn_rate': self.data['is_churned'].mean(),
            'avg_gmv': self.data['total_gmv'].mean() if 'total_gmv' in self.data.columns else None,
            'avg_orders': self.data['total_orders'].mean() if 'total_orders' in self.data.columns else None
        }
        
        # Segment analysis
        if 'segment' in self.data.columns:
            segment_insights = self.data.groupby('segment').agg({
                'is_churned': ['count', 'mean'],
                'total_gmv': 'mean'
            }).round(3)
            insights['segment_analysis'] = segment_insights.to_dict()
        
        # Risk factors
        if 'days_since_last_order' in self.data.columns:
            high_risk_customers = self.data[self.data['days_since_last_order'] > 60]
            insights['risk_factors'] = {
                'customers_inactive_60_days': len(high_risk_customers),
                'churn_rate_inactive_customers': high_risk_customers['is_churned'].mean()
            }
        
        # Model performance (if predictions available)
        if predictions_df is not None and 'churn_probability' in predictions_df.columns:
            from sklearn.metrics import roc_auc_score, precision_score, recall_score
            
            y_true = predictions_df['actual_churn']
            y_prob = predictions_df['churn_probability']
            y_pred = (y_prob >= 0.5).astype(int)
            
            insights['model_performance'] = {
                'auc_score': roc_auc_score(y_true, y_prob),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred)
            }
        
        logger.info("Generated comprehensive insights report")
        return insights 