"""
General CSV Data Cleaning Module for Data Science Projects

This module accepts a CSV file, loads the data, performs general
level data cleaning, and outputs a new CSV file with the cleaned data.

Usage:
    python general_csv_data_cleaning.py <input_csv_file> [--output <output_csv_file>]
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple


class CsvDataCleaner:
    """
    A class to perform general level data cleaning operations on datasets
    loaded from CSV files and export cleaned data as a new CSV file.
    """

    def __init__(self):
        """
        Initialize the CsvDataCleaner.
        """
        self.df = None
        self.table_name = "cleaned_data"
        self.cleaning_report = {}

    def load_from_csv_file(
        self,
        csv_file_path: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            csv_file_path: Path to the CSV file to load.
            **kwargs: Additional arguments to pass to pd.read_csv().

        Returns:
            DataFrame containing the loaded data.
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        self.df = pd.read_csv(csv_file_path, **kwargs)
        self.table_name = os.path.splitext(os.path.basename(csv_file_path))[0]

        self.cleaning_report = {
            'input_file': csv_file_path,
            'initial_rows': len(self.df),
            'initial_columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'steps_applied': []
        }

        print(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns from CSV file")
        return self.df

    # =========================================================================
    # MISSING VALUE HANDLING
    # =========================================================================

    def handle_missing_values(self, strategy: str = 'auto') -> 'CsvDataCleaner':
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
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col] = self.df[col].fillna(mode_val[0])
                    else:
                        self.df[col] = self.df[col].fillna('UNKNOWN')
        elif strategy == 'drop':
            self.df = self.df.dropna()

        final_missing = self.df.isnull().sum().sum()
        self._log_step(f'Handled missing values: {initial_missing} -> {final_missing}')
        return self

    # =========================================================================
    # DUPLICATE HANDLING
    # =========================================================================

    def handle_duplicates(self, keep: str = 'first') -> 'CsvDataCleaner':
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
    ) -> 'CsvDataCleaner':
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

    def clean_strings(self) -> 'CsvDataCleaner':
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

    def convert_dtypes(self) -> 'CsvDataCleaner':
        """Automatically infer and convert column data types."""
        if self.df is None:
            raise ValueError("No data loaded.")

        self.df = self.df.infer_objects()
        self._log_step('Converted data types (auto-inferred)')
        return self

    # =========================================================================
    # COLUMN OPERATIONS
    # =========================================================================

    def rename_columns(self) -> 'CsvDataCleaner':
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

    def drop_constant_columns(self, threshold: float = 0.01) -> 'CsvDataCleaner':
        """Drop columns with little to no variance."""
        if self.df is None:
            raise ValueError("No data loaded.")

        cols_to_drop = []
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio < threshold:
                cols_to_drop.append(col)

        self.df = self.df.drop(columns=cols_to_drop)
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
    ) -> 'CsvDataCleaner':
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
    # CSV EXPORT
    # =========================================================================

    def export_to_csv(
        self,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export cleaned data as a CSV file.

        Args:
            output_path: Path for output CSV file. If None, generates timestamp-based name.
            **kwargs: Additional arguments to pass to df.to_csv().

        Returns:
            Path to the output CSV file.
        """
        if self.df is None:
            raise ValueError("No data to export.")

        # Generate output filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{self.table_name}_cleaned_{timestamp}.csv"

        # Write the DataFrame to CSV
        self.df.to_csv(output_path, index=False, **kwargs)

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