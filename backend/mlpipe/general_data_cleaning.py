"""
General Data Cleaning Module for Data Science Projects

This module accepts a SQL file, executes it to fetch data, performs general
level data cleaning, and outputs a new SQL file with INSERT statements
for the cleaned data.

Usage:
    python general_data_cleaning.py <input_sql_file> [--output <output_sql_file>]
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple


class DataCleaner:
    """
    A class to perform general level data cleaning operations on datasets
    loaded from SQL databases and export cleaned data as SQL INSERT statements.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the DataCleaner.

        Args:
            connection_string: SQL database connection string.
        """
        self.connection_string = connection_string
        self.engine = None
        self.df = None
        self.table_name = "cleaned_data"
        self.cleaning_report = {}

    def connect(self, connection_string: str) -> None:
        """Establish connection to the SQL database."""
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)

    def load_from_sql_file(self, sql_file_path: str) -> pd.DataFrame:
        """
        Load data by executing SQL from a file.

        Args:
            sql_file_path: Path to the SQL file to execute.

        Returns:
            DataFrame containing the fetched data.
        """
        if not os.path.exists(sql_file_path):
            raise FileNotFoundError(f"SQL file not found: {sql_file_path}")

        with open(sql_file_path, 'r', encoding='utf-8') as f:
            query = f.read()

        if not self.engine:
            raise ValueError("Database connection not established. Call connect() first.")

        self.df = pd.read_sql_query(query, self.engine)

        # Extract table name from SQL file name for output
        base_name = os.path.basename(sql_file_path)
        self.table_name = os.path.splitext(base_name)[0]

        self.cleaning_report = {
            'input_file': sql_file_path,
            'initial_rows': len(self.df),
            'initial_columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'steps_applied': []
        }

        print(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns from SQL file")
        return self.df

    # =========================================================================
    # MISSING VALUE HANDLING
    # =========================================================================

    def handle_missing_values(self, strategy: str = 'auto') -> 'DataCleaner':
        """
        Handle missing values in the dataset.

        Args:
            strategy: 'drop', 'fill', 'auto'
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        initial_missing = self.df.isnull().sum().sum()

        if strategy == 'auto':
            for col in self.df.columns:
                if self._is_numeric(col):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                    else:
                        self.df[col].fillna('UNKNOWN', inplace=True)
        elif strategy == 'drop':
            self.df.dropna(inplace=True)

        final_missing = self.df.isnull().sum().sum()
        self._log_step(f'Handled missing values: {initial_missing} -> {final_missing}')
        return self

    # =========================================================================
    # DUPLICATE HANDLING
    # =========================================================================

    def handle_duplicates(self, keep: str = 'first') -> 'DataCleaner':
        """Remove duplicate rows."""
        if self.df is None:
            raise ValueError("No data loaded.")

        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(keep=keep)
        removed = initial_count - len(self.df)
        self._log_step(f'Removed {removed} duplicate rows')
        return self

    # =========================================================================
    # OUTLIER HANDLING
    # =========================================================================

    def handle_outliers(
        self,
        method: str = 'iqr',
        iqr_multiplier: float = 1.5,
        strategy: str = 'cap'
    ) -> 'DataCleaner':
        """
        Handle outliers in numeric columns.

        Args:
            method: 'iqr' or 'zscore'
            strategy: 'remove', 'cap'
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - iqr_multiplier * IQR
                upper = Q3 + iqr_multiplier * IQR

                if strategy == 'cap':
                    self.df[col] = self.df[col].clip(lower=lower, upper=upper)

        self._log_step(f'Handled outliers in {len(numeric_cols)} columns (method={method})')
        return self

    # =========================================================================
    # STRING CLEANING
    # =========================================================================

    def clean_strings(self) -> 'DataCleaner':
        """Clean string columns with common text operations."""
        if self.df is None:
            raise ValueError("No data loaded.")

        string_cols = self.df.select_dtypes(include=['object', 'string']).columns

        for col in string_cols:
            # Strip whitespace
            self.df[col] = self.df[col].astype(str).str.strip()
            # Replace multiple spaces with single
            self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
            # Standardize null representations
            null_patterns = ['null', 'none', 'na', 'n/a', '', '-', '.']
            mask = self.df[col].str.lower().isin(null_patterns)
            self.df.loc[mask, col] = np.nan

        self._log_step(f'Cleaned strings in {len(string_cols)} columns')
        return self

    # =========================================================================
    # DATA TYPE CONVERSION
    # =========================================================================

    def convert_dtypes(self) -> 'DataCleaner':
        """Automatically infer and convert column data types."""
        if self.df is None:
            raise ValueError("No data loaded.")

        self.df = self.df.infer_objects()
        self._log_step('Converted data types (auto-inferred)')
        return self

    # =========================================================================
    # COLUMN OPERATIONS
    # =========================================================================

    def rename_columns(self) -> 'DataCleaner':
        """Rename columns to lowercase with underscores."""
        if self.df is None:
            raise ValueError("No data loaded.")

        new_columns = []
        for col in self.df.columns:
            new_name = col.lower().strip().replace(' ', '_').replace('-', '_')
            new_name = re.sub(r'[^a-z0-9_]', '', new_name)
            new_columns.append(new_name)

        self.df.columns = new_columns
        self._log_step('Renamed columns to lowercase with underscores')
        return self

    def drop_constant_columns(self, threshold: float = 0.99) -> 'DataCleaner':
        """Drop columns with little to no variance."""
        if self.df is None:
            raise ValueError("No data loaded.")

        cols_to_drop = []
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio < threshold:
                cols_to_drop.append(col)

        self.df.drop(columns=cols_to_drop, inplace=True)
        self._log_step(f'Dropped {len(cols_to_drop)} constant columns')
        return self

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _is_numeric(self, column: str) -> bool:
        """Check if a column is numeric."""
        return pd.api.types.is_numeric_dtype(self.df[column])

    def _log_step(self, step: str) -> None:
        """Log a cleaning step."""
        self.cleaning_report['steps_applied'].append(step)

    def full_clean(
        self,
        handle_outliers: bool = True,
        outlier_method: str = 'iqr',
        outlier_multiplier: float = 1.5
    ) -> 'DataCleaner':
        """
        Apply the standard full cleaning pipeline.

        Args:
            handle_outliers: Whether to handle outliers.
            outlier_method: 'iqr' or 'zscore'.
            outlier_multiplier: IQR multiplier or z-score threshold.
        """
        print("\nApplying full cleaning pipeline...")

        self.handle_missing_values(strategy='auto')
        self.handle_duplicates()
        self.clean_strings()
        self.convert_dtypes()
        self.drop_constant_columns()

        if handle_outliers:
            self.handle_outliers(
                method=outlier_method,
                iqr_multiplier=outlier_multiplier,
                strategy='cap'
            )

        self.rename_columns()

        print("Full cleaning pipeline complete.\n")
        return self

    # =========================================================================
    # SQL EXPORT
    # =========================================================================

    def export_to_sql(
        self,
        output_path: Optional[str] = None,
        chunk_size: int = 1000
    ) -> str:
        """
        Export cleaned data as SQL INSERT statements.

        Args:
            output_path: Path for output SQL file. If None, generates timestamp-based name.
            chunk_size: Number of rows per INSERT statement batch.

        Returns:
            Path to the output SQL file.
        """
        if self.df is None:
            raise ValueError("No data to export.")

        # Generate output filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{self.table_name}_cleaned_{timestamp}.sql"

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("-- Cleaned Data Export\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n")
            f.write(f"-- Source: {self.cleaning_report.get('input_file', 'Unknown')}\n")
            f.write(f"-- Rows: {len(self.df)}, Columns: {len(self.df.columns)}\n")
            f.write("--\n\n")

            # Write cleaning report
            f.write("-- Cleaning Steps Applied:\n")
            for i, step in enumerate(self.cleaning_report.get('steps_applied', []), 1):
                f.write(f"-- {i}. {step}\n")
            f.write("\n")

            # Create table statement
            f.write(f"-- Create table statement\n")
            f.write(f"DROP TABLE IF EXISTS `{self.table_name}_cleaned`;\n")
            f.write(f"CREATE TABLE `{self.table_name}_cleaned` (\n")

            for col in self.df.columns:
                dtype = self.df[col].dtype
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "REAL"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sql_type = "DATETIME"
                else:
                    sql_type = "TEXT"
                f.write(f"    `{col}` {sql_type},\n")

            # Remove last comma and close
            f.seek(f.tell() - 2)
            f.write("\n);\n\n")

            # Insert statements
            f.write("-- Insert statements\n")

            for i in range(0, len(self.df), chunk_size):
                chunk = self.df.iloc[i:i+chunk_size]
                values_list = []

                for _, row in chunk.iterrows():
                    values = []
                    for col in self.df.columns:
                        val = row[col]
                        if pd.isna(val):
                            values.append("NULL")
                        elif isinstance(val, (int, float)):
                            values.append(str(val))
                        elif isinstance(val, datetime):
                            values.append(f"'{val.isoformat()}'")
                        else:
                            # Escape single quotes in strings
                            escaped = str(val).replace("'", "''")
                            values.append(f"'{escaped}'")
                    values_list.append(f"({', '.join(values)})")

                if values_list:
                    cols = ', '.join([f"`{col}`" for col in self.df.columns])
                    f.write(f"INSERT INTO `{self.table_name}_cleaned` ({cols}) VALUES\n")
                    f.write(",\n".join(values_list))
                    f.write(";\n\n")

        print(f"Exported cleaned data to: {output_path}")
        return output_path

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get the cleaning report."""
        if self.df is not None:
            self.cleaning_report['final_rows'] = len(self.df)
            self.cleaning_report['final_columns'] = len(self.df.columns)
            self.cleaning_report['missing_values'] = {
                str(k): int(v) for k, v in self.df.isnull().sum().to_dict().items()
            }
        return self.cleaning_report

    def get_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        return self.df


# Note: This module is designed to be imported by pipeline.py
# which orchestrates the full data processing workflow.
