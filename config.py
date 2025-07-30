import os
from typing import Optional

class Config:
    """Configuration class for the churn prediction project."""
    
    # Neon Database Configuration (Hardcoded for direct script execution)
    NEON_DB_HOST: str = "ep-holy-water-a12w1h8d-pooler.ap-southeast-1.aws.neon.tech"
    NEON_DB_NAME: str = "neondb"
    NEON_DB_USER: str = "neondb_owner"
    NEON_DB_PASSWORD: str = "npg_w1NLHMyxSRg7"
    NEON_DB_PORT: int = 5432
    NEON_DB_SSLMODE: str = "require"
    
    # Model Configuration
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    
    # Feature Engineering Configuration
    CHURN_THRESHOLD: float = 0.5  # 50% reduction threshold
    HISTORY_DAYS: int = 365
    CURRENT_PERIOD_DAYS: int = 60
    PREVIOUS_PERIOD_DAYS: int = 60
    BASELINE_PERIOD_DAYS: int = 120
    
    @classmethod
    def get_db_connection_string(cls) -> str:
        """Generate database connection string."""
        return (
            f"postgresql://{cls.NEON_DB_USER}:{cls.NEON_DB_PASSWORD}"
            f"@{cls.NEON_DB_HOST}:{cls.NEON_DB_PORT}/{cls.NEON_DB_NAME}"
            f"?sslmode={cls.NEON_DB_SSLMODE}"
        )
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present."""
        required_fields = [
            cls.NEON_DB_HOST, cls.NEON_DB_NAME, 
            cls.NEON_DB_USER, cls.NEON_DB_PASSWORD
        ]
        
        return all(
            field and not field.startswith('your-')
            for field in required_fields
        ) 