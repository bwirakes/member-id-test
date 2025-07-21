import psycopg2
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeonDBConnector:
    """
    A robust database connector for Neon PostgreSQL database.
    Handles connections, query execution, and data retrieval for the churn prediction project.
    """
    
    def __init__(self, config: Config = None):
        """Initialize the database connector with configuration."""
        self.config = config or Config()
        self.connection_params = {
            'host': self.config.NEON_DB_HOST,
            'database': self.config.NEON_DB_NAME,
            'user': self.config.NEON_DB_USER,
            'password': self.config.NEON_DB_PASSWORD,
            'port': self.config.NEON_DB_PORT,
            'sslmode': self.config.NEON_DB_SSLMODE,
            'connect_timeout': 30
        }
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup."""
        connection = None
        try:
            connection = psycopg2.connect(**self.connection_params)
            yield connection
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    logger.info("Database connection successful")
                    return result[0] == 1
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries
            
        Returns:
            pandas.DataFrame: Query results
        """
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"Query executed successfully. Returned {len(df)} rows.")
                return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get information about a table's structure."""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = %s
        ORDER BY ordinal_position;
        """
        return self.execute_query(query, (table_name,))
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        result = self.execute_query(query)
        return result.iloc[0]['row_count']
    
    def get_date_range(self, table_name: str, date_column: str) -> Dict[str, Any]:
        """Get the date range for a specific table and date column."""
        query = f"""
        SELECT 
            MIN({date_column}) as min_date,
            MAX({date_column}) as max_date,
            COUNT(DISTINCT {date_column}) as unique_dates
        FROM {table_name}
        WHERE {date_column} IS NOT NULL
        """
        result = self.execute_query(query)
        return result.iloc[0].to_dict()
    
    def create_churn_labels(self) -> pd.DataFrame:
        """
        Create churn labels based on spend reduction patterns.
        Implements the provided SQL logic for target variable creation.
        """
        query = """
        WITH customer_spend_periods AS (
            SELECT 
                member_id,
                -- Current period (last 60 days)
                SUM(CASE 
                    WHEN order_date::date >= CURRENT_DATE - INTERVAL '60 days' 
                    THEN grand_total ELSE 0 
                END) as spend_last_60_days,
                
                -- Previous period (61-120 days ago)
                SUM(CASE 
                    WHEN order_date::date >= CURRENT_DATE - INTERVAL '120 days' 
                    AND order_date::date < CURRENT_DATE - INTERVAL '60 days'
                    THEN grand_total ELSE 0 
                END) as spend_previous_60_days,
                
                -- Historical baseline (average monthly spend * 2)
                AVG(CASE 
                    WHEN order_date::date < CURRENT_DATE - INTERVAL '120 days'
                    THEN grand_total ELSE NULL 
                END) * 2 as historical_60day_baseline
            FROM order_header
            WHERE order_date::date >= CURRENT_DATE - INTERVAL '365 days' -- Need sufficient history
            GROUP BY member_id
            HAVING COUNT(*) >= 3 -- Ensure sufficient transaction history
        ),
        churn_labels AS (
            SELECT 
                member_id,
                spend_last_60_days,
                spend_previous_60_days,
                historical_60day_baseline,
                
                -- Multiple churn definitions for robustness
                CASE 
                    WHEN spend_previous_60_days > 0 
                    AND spend_last_60_days <= spend_previous_60_days * 0.5 
                    THEN 1 ELSE 0 
                END as churned_vs_previous,
                
                CASE 
                    WHEN historical_60day_baseline > 0 
                    AND spend_last_60_days <= historical_60day_baseline * 0.5 
                    THEN 1 ELSE 0 
                END as churned_vs_baseline,
                
                -- Final churn label (either method)
                CASE 
                    WHEN (spend_previous_60_days > 0 AND spend_last_60_days <= spend_previous_60_days * 0.5)
                    OR (historical_60day_baseline > 0 AND spend_last_60_days <= historical_60day_baseline * 0.5)
                    THEN 1 ELSE 0 
                END as is_churned
            FROM customer_spend_periods
        )
        SELECT * FROM churn_labels
        """
        
        logger.info("Creating churn labels...")
        return self.execute_query(query)
    
    def create_feature_set(self) -> pd.DataFrame:
        """
        Create comprehensive feature set for churn prediction.
        Implements the provided SQL logic for feature engineering.
        """
        query = """
        WITH base_features AS (
            SELECT 
                oh.member_id,
                
                -- Recency features
                CURRENT_DATE - MAX(oh.order_date::date) as days_since_last_order,
                CURRENT_DATE - MIN(oh.order_date::date) as customer_age_days,
                
                -- Frequency features
                COUNT(DISTINCT oh.id) as total_orders,
                COUNT(DISTINCT DATE_TRUNC('month', oh.order_date)) as active_months,
                ROUND(COUNT(DISTINCT oh.id)::numeric / 
                      NULLIF(COUNT(DISTINCT DATE_TRUNC('month', oh.order_date)), 0), 2) as avg_orders_per_month,
                
                -- Monetary features
                SUM(oh.grand_total) as total_gmv,
                AVG(oh.grand_total) as avg_order_value,
                STDDEV(oh.grand_total) as order_value_std,
                
                -- Trend features (last 3 months vs previous 3 months)
                SUM(CASE 
                    WHEN oh.order_date::date >= CURRENT_DATE - INTERVAL '90 days' 
                    THEN oh.grand_total ELSE 0 
                END) as gmv_last_90_days,
                
                SUM(CASE 
                    WHEN oh.order_date::date >= CURRENT_DATE - INTERVAL '180 days' 
                    AND oh.order_date::date < CURRENT_DATE - INTERVAL '90 days'
                    THEN oh.grand_total ELSE 0 
                END) as gmv_previous_90_days,
                
                -- Channel behavior
                COUNT(DISTINCT oh.outlet_name) as outlet_diversity,
                MODE() WITHIN GROUP (ORDER BY oh.outlet_name) as primary_outlet,
                
                -- Membership features
                oh.member_tier_when_transact as current_tier
                
            FROM order_header oh
            WHERE oh.order_date::date >= CURRENT_DATE - INTERVAL '365 days'
            GROUP BY oh.member_id, oh.member_tier_when_transact
        ),
        product_features AS (
            SELECT 
                oh.member_id,
                
                -- Product diversity
                COUNT(DISTINCT oi.product_group) as product_group_diversity,
                COUNT(DISTINCT oi.brand_name) as brand_diversity,
                COUNT(DISTINCT oi.sku) as sku_diversity,
                
                -- Purchase patterns
                AVG(oi.quantity) as avg_quantity_per_item,
                SUM(oi.quantity * oi.paid_price) / NULLIF(SUM(oi.quantity * oi.price), 0) as avg_discount_rate,
                
                -- Category affinity (top 3 product groups)
                STRING_AGG(DISTINCT oi.product_group, '|' ORDER BY COUNT(*) DESC) as top_product_groups
                
            FROM order_header oh
            JOIN order_item oi ON oh.order_number = oi.order_number
            WHERE oh.order_date::date >= CURRENT_DATE - INTERVAL '365 days'
            GROUP BY oh.member_id
        ),
        behavioral_features AS (
            SELECT 
                member_id,
                
                -- Seasonality patterns
                MODE() WITHIN GROUP (ORDER BY EXTRACT(DOW FROM order_date)) as favorite_day_of_week,
                AVG(EXTRACT(DOW FROM order_date)) as avg_day_of_week,
                
                -- Order timing patterns
                AVG(grand_total) FILTER (WHERE EXTRACT(DOW FROM order_date) IN (6,0)) as weekend_avg_spend,
                AVG(grand_total) FILTER (WHERE EXTRACT(DOW FROM order_date) BETWEEN 1 AND 5) as weekday_avg_spend,
                
                -- Inter-purchase time
                AVG(days_between_orders) as avg_days_between_orders,
                STDDEV(days_between_orders) as days_between_orders_std
                
            FROM (
                SELECT 
                    member_id,
                    order_date,
                    grand_total,
                    order_date::date - LAG(order_date::date) OVER (PARTITION BY member_id ORDER BY order_date) as days_between_orders
                FROM order_header
                WHERE order_date::date >= CURRENT_DATE - INTERVAL '365 days'
            ) t
            GROUP BY member_id
        ),
        final_features AS (
            SELECT 
                bf.*,
                pf.product_group_diversity,
                pf.brand_diversity,
                pf.sku_diversity,
                pf.avg_quantity_per_item,
                pf.avg_discount_rate,
                pf.top_product_groups,
                
                bhf.avg_days_between_orders,
                bhf.days_between_orders_std,
                bhf.weekend_avg_spend,
                bhf.weekday_avg_spend,
                
                -- Derived features
                CASE 
                    WHEN bf.gmv_previous_90_days > 0 
                    THEN (bf.gmv_last_90_days - bf.gmv_previous_90_days) / bf.gmv_previous_90_days 
                    ELSE 0 
                END as spend_growth_rate,
                
                bf.avg_order_value / NULLIF(bf.total_gmv::numeric / bf.customer_age_days * 365, 0) as aov_vs_annual_spend_ratio,
                
                CASE 
                    WHEN bf.days_since_last_order <= 7 THEN 'very_recent'
                    WHEN bf.days_since_last_order <= 30 THEN 'recent' 
                    WHEN bf.days_since_last_order <= 60 THEN 'moderate'
                    ELSE 'old'
                END as recency_bucket
                
            FROM base_features bf
            LEFT JOIN product_features pf ON bf.member_id = pf.member_id
            LEFT JOIN behavioral_features bhf ON bf.member_id = bhf.member_id
        )
        SELECT * FROM final_features
        """
        
        logger.info("Creating feature set...")
        return self.execute_query(query) 