import os
from typing import Optional

class Config:
    """Configuration class for the churn prediction project."""
    
    # Neon Database Configuration
    NEON_DB_HOST: str = os.getenv('NEON_DB_HOST', 'your-neon-hostname.neon.tech')
    NEON_DB_NAME: str = os.getenv('NEON_DB_NAME', 'your-database-name')
    NEON_DB_USER: str = os.getenv('NEON_DB_USER', 'your-username')
    NEON_DB_PASSWORD: str = os.getenv('NEON_DB_PASSWORD', 'your-password')
    NEON_DB_PORT: int = int(os.getenv('NEON_DB_PORT', '5432'))
    NEON_DB_SSLMODE: str = os.getenv('NEON_DB_SSLMODE', 'require')
    
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
            field and field != f'your-{field.split("_")[-1].lower()}' 
            for field in required_fields
        ) 