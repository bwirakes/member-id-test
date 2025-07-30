#!/usr/bin/env python3
"""
Insert Order Item Data from Excel to Neon Database

This script reads data from an Excel file and inserts it into the 'order_item' table
in a Neon PostgreSQL database. It's designed to process multiple sheets from the
Excel file, skipping the first sheet as per the user's request.

Usage:
    python insert_order_items.py /path/to/your/order_item.xls

Prerequisites:
    - pandas
    - openpyxl (for .xlsx/.xls support)
    - psycopg2-binary
    - sqlalchemy

You can install these using pip:
    pip install pandas openpyxl psycopg2-binary sqlalchemy
"""

import os
import sys
import logging
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# The user provided a full connection string.
# For security, it's better to load this from an environment variable or a config file.
# However, for this specific request, I will use the one provided.
NEON_DB_CONNECTION_STRING = "postgresql://neondb_owner:npg_w1NLHMyxSRg7@ep-holy-water-a12w1h8d-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
TABLE_NAME = 'order_item'

def get_db_engine(connection_string: str):
    """Creates a SQLAlchemy engine from a connection string."""
    try:
        # The provided connection string has a 'channel_binding' parameter that is not
        # supported by SQLAlchemy's postgresql dialect. I will remove it.
        parsed_url = urlparse(connection_string)
        query_params = parse_qs(parsed_url.query)
        
        # Remove the unsupported parameter
        if 'channel_binding' in query_params:
            del query_params['channel_binding']
            
        # Rebuild the query string
        new_query = '&'.join([f"{k}={v[0]}" for k, v in query_params.items()])
        
        # Reconstruct the URL without the unsupported parameter
        db_url = parsed_url._replace(query=new_query).geturl()
        
        engine = create_engine(db_url)
        logger.info("Database engine created successfully.")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise

def read_excel_sheets(file_path: str) -> dict[str, pd.DataFrame]:
    """Reads all sheets from an Excel file, skipping the first one."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found at: {file_path}")
            return {}

        logger.info(f"Reading Excel file from: {file_path}")
        # Use None to get all sheets, then we'll skip the first one
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        
        sheet_names = list(all_sheets.keys())
        if len(sheet_names) <= 1:
            logger.warning("No subsequent sheets to process after the first one.")
            return {}
            
        sheets_to_process = {name: df for name, df in all_sheets.items() if name != sheet_names[0]}
        
        logger.info(f"Found {len(sheets_to_process)} sheets to process: {list(sheets_to_process.keys())}")
        return sheets_to_process
        
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        raise

def insert_data_to_db(engine, data_frames: dict[str, pd.DataFrame], table_name: str):
    """Inserts data from a dictionary of DataFrames into a database table."""
    total_rows_inserted = 0
    for sheet_name, df in data_frames.items():
        logger.info(f"Processing sheet: {sheet_name} with {len(df)} rows.")
        try:
            # It's good practice to ensure columns match the target table.
            # For this script, we assume the Excel sheets have the correct columns.
            # A more robust solution would fetch table schema and align columns.
            
            df.to_sql(table_name, engine, if_exists='append', index=False)
            total_rows_inserted += len(df)
            logger.info(f"Successfully inserted {len(df)} rows from sheet '{sheet_name}'.")
        
        except Exception as e:
            logger.error(f"Failed to insert data from sheet '{sheet_name}': {e}")
            logger.error("Skipping this sheet and continuing with others.")
            continue
            
    return total_rows_inserted

def main():
    """Main function to orchestrate the data insertion process."""
    if len(sys.argv) < 2:
        logger.error("Usage: python insert_order_items.py /path/to/your/order_item.xls")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    try:
        # Get DB engine
        engine = get_db_engine(NEON_DB_CONNECTION_STRING)
        
        # Read Excel file
        sheets = read_excel_sheets(file_path)
        
        if not sheets:
            logger.info("No data to insert. Exiting.")
            sys.exit(0)
            
        # Insert data
        rows_inserted = insert_data_to_db(engine, sheets, TABLE_NAME)
        
        logger.info(f"\n--- Process Complete ---")
        logger.info(f"Total rows inserted into '{TABLE_NAME}': {rows_inserted}")
        logger.info("------------------------")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during the pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 