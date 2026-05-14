"""
Data Ingestion Layer
Loads, validates, and prepares test execution logs
"""

import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionLayer:
    """
    Handles data ingestion from CSV files containing test execution history.
    
    Attributes:
        data_source (str): Path to the CSV file
        raw_logs (pd.DataFrame): Loaded raw logs
        change_metadata (dict): Test ID to change frequency mapping
    """
    
    REQUIRED_COLUMNS = {
        "test_id": "object",
        "test_name": "object",
        "test_suite": "object",
        "cycle_id": "int64",
        "outcome": "int64",
        "execution_time": "float64",
        "change_frequency": "int64",
        "last_result": "int64",
        "days_since_last_failure": "int64"
    }
    
    def __init__(self, data_source: str):
        """
        Initialize the DataIngestionLayer.
        
        Args:
            data_source (str): Path to CSV file
        """
        self.data_source = Path(data_source)
        self.raw_logs = None
        self.change_metadata = {}
        logger.info(f"[DataIngestion] Initialized with source: {self.data_source}")
    
    def load_historical_logs(self) -> pd.DataFrame:
        """
        Load test execution logs from CSV.
        
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If file is empty
        """
        if not self.data_source.exists():
            raise FileNotFoundError(f"Data source not found: {self.data_source}")
        
        self.raw_logs = pd.read_csv(self.data_source)
        
        if len(self.raw_logs) == 0:
            raise ValueError(f"CSV file is empty: {self.data_source}")
        
        logger.info(f"[DataIngestion] Loaded {len(self.raw_logs)} records from {self.data_source}")
        return self.raw_logs
    
    def load_change_metadata(self) -> dict:
        """
        Extract change frequency metadata per test.
        
        Returns:
            dict: Mapping of test_id -> change_frequency
        """
        if self.raw_logs is None:
            self.load_historical_logs()
        
        # For each test_id, use the most recent change_frequency value
        self.change_metadata = (
            self.raw_logs
            .sort_values("cycle_id", ascending=False)
            .drop_duplicates(subset="test_id", keep="first")
            .set_index("test_id")["change_frequency"]
            .to_dict()
        )
        
        logger.info(f"[DataIngestion] Loaded change metadata for {len(self.change_metadata)} tests")
        return self.change_metadata
    
    def validate_schema(self) -> bool:
        """
        Validate that raw logs have the required schema.
        
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If schema is invalid
        """
        if self.raw_logs is None:
            self.load_historical_logs()
        
        # Check required columns exist
        missing_cols = set(self.REQUIRED_COLUMNS.keys()) - set(self.raw_logs.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Expected: {set(self.REQUIRED_COLUMNS.keys())}\n"
                f"Found: {set(self.raw_logs.columns)}"
            )
        
        # Check data types
        for col, expected_dtype in self.REQUIRED_COLUMNS.items():
            actual_dtype = str(self.raw_logs[col].dtype)
            if actual_dtype not in [expected_dtype, "int64", "float64", "object"]:  # Allow some flexibility
                logger.warning(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
        
        # Check outcome is binary (0/1)
        unique_outcomes = self.raw_logs["outcome"].unique()
        if not all(x in [0, 1] for x in unique_outcomes):
            raise ValueError(f"'outcome' column must be binary (0/1), found: {unique_outcomes}")
        
        logger.info("[DataIngestion] Schema validation passed ✓")
        return True
    
    def preprocess(self):
        """
        Load, validate, and prepare data (entry point for preprocessing module).
        
        Returns:
            pd.DataFrame: Cleaned and validated data
        """
        # Import here to avoid circular imports
        from preprocessing import clean_data, encode_labels
        
        self.load_historical_logs()
        self.validate_schema()
        self.load_change_metadata()
        
        # Apply preprocessing
        df = clean_data(self.raw_logs)
        df = encode_labels(df)
        
        logger.info("[DataIngestion] Preprocessing complete")
        return df