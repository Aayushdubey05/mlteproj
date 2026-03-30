"""
Test suite for general_data_cleaning.py module.

Tests cover:
1. Missing value handling
2. Duplicate removal
3. Outlier handling
4. String cleaning
5. Data type conversion
6. Column operations
7. SQL export functionality
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from general_data_cleaning import DataCleaner


class TestDataCleanerInitialization:
    """Test DataCleaner initialization and connection."""

    def test_init_without_connection(self):
        """Test initialization without connection string."""
        cleaner = DataCleaner()
        assert cleaner.connection_string is None
        assert cleaner.engine is None
        assert cleaner.df is None

    def test_init_with_connection(self):
        """Test initialization with connection string."""
        cleaner = DataCleaner("sqlite:///test.db")
        assert cleaner.connection_string == "sqlite:///test.db"

    def test_connect(self):
        """Test connect method."""
        cleaner = DataCleaner()
        cleaner.connect("sqlite:///test.db")
        assert cleaner.connection_string == "sqlite:///test.db"
        assert cleaner.engine is not None


class TestLoadFromSQLFile:
    """Test loading data from SQL files."""

    def test_sql_file_not_found(self):
        """Test error when SQL file doesn't exist."""
        cleaner = DataCleaner("sqlite:///test.db")
        with pytest.raises(FileNotFoundError):
            cleaner.load_from_sql_file("nonexistent.sql")

    def test_no_connection_error(self):
        """Test error when no connection established."""
        cleaner = DataCleaner()
        # Create a temp SQL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write("SELECT 1")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="connection"):
                cleaner.load_from_sql_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestMissingValueHandling:
    """Test missing value handling functionality."""

    def test_auto_strategy_numeric(self):
        """Test auto strategy fills numeric with median."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'num_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        cleaner.handle_missing_values(strategy='auto')
        assert cleaner.df['num_col'].isnull().sum() == 0
        assert cleaner.df['num_col'].iloc[2] == 3.0  # median

    def test_auto_strategy_categorical(self):
        """Test auto strategy fills categorical with mode."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'num_col': [1, 2, 3, 4, 5],
            'str_col': ['a', 'a', np.nan, 'b', 'c']
        })
        cleaner.handle_missing_values(strategy='auto')
        assert cleaner.df['str_col'].isnull().sum() == 0
        assert cleaner.df['str_col'].iloc[2] == 'a'  # mode

    def test_drop_strategy(self):
        """Test drop strategy removes rows with missing values."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': ['a', np.nan, 'c', 'd']
        })
        initial_rows = len(cleaner.df)
        cleaner.handle_missing_values(strategy='drop')
        assert len(cleaner.df) < initial_rows
        assert cleaner.df.isnull().sum().sum() == 0

    def test_no_data_loaded_error(self):
        """Test error when no data is loaded."""
        cleaner = DataCleaner()
        with pytest.raises(ValueError, match="No data loaded"):
            cleaner.handle_missing_values()


class TestDuplicateHandling:
    """Test duplicate removal functionality."""

    def test_remove_duplicates_keep_first(self):
        """Test removing duplicates keeps first occurrence."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': [1, 1, 2, 3],
            'col2': ['a', 'a', 'b', 'c']
        })
        initial_rows = len(cleaner.df)
        cleaner.handle_duplicates(keep='first')
        assert len(cleaner.df) == 3
        assert cleaner.df.iloc[0]['col1'] == 1

    def test_remove_duplicates_keep_last(self):
        """Test removing duplicates keeps last occurrence."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': [1, 1, 2, 3],
            'col2': ['a', 'a', 'b', 'c']
        })
        cleaner.handle_duplicates(keep='last')
        assert len(cleaner.df) == 3
        assert cleaner.df.iloc[0]['col1'] == 1

    def test_remove_duplicates_drop_all(self):
        """Test removing duplicates drops all duplicates."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': [1, 1, 2, 3],
            'col2': ['a', 'a', 'b', 'c']
        })
        cleaner.handle_duplicates(keep=False)
        assert len(cleaner.df) == 2  # Only unique rows remain

    def test_duplicate_tracking(self):
        """Test that removed duplicates are tracked in report."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': [1, 1, 2],
            'col2': ['a', 'a', 'b']
        })
        cleaner.cleaning_report = {'steps_applied': []}
        cleaner.handle_duplicates()
        assert any('duplicate' in step for step in cleaner.cleaning_report['steps_applied'])


class TestOutlierHandling:
    """Test outlier handling functionality."""

    def test_iqr_method_cap(self):
        """Test IQR method with capping strategy."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        cleaner.handle_outliers(method='iqr', iqr_multiplier=1.5, strategy='cap')
        # The outlier should be capped
        assert cleaner.df['values'].max() < 100

    def test_iqr_method_no_outliers(self):
        """Test IQR method when no outliers exist."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5]
        })
        original_max = cleaner.df['values'].max()
        cleaner.handle_outliers(method='iqr', strategy='cap')
        assert cleaner.df['values'].max() == original_max

    def test_zscore_method(self):
        """Test z-score method for outlier detection."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 50]  # 50 is an outlier
        })
        cleaner.handle_outliers(method='zscore', zscore_threshold=2.0, strategy='cap')
        assert cleaner.df['values'].max() < 50

    def test_multiple_numeric_columns(self):
        """Test outlier handling on multiple numeric columns."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 100],
            'col2': [10, 20, 30, 40, 500],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        cleaner.handle_outliers(method='iqr', strategy='cap')
        # Both numeric columns should be processed
        assert cleaner.df['col1'].max() < 100
        assert cleaner.df['col2'].max() < 500


class TestStringCleaning:
    """Test string cleaning functionality."""

    def test_strip_whitespace(self):
        """Test stripping whitespace from strings."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'text': ['  hello  ', '  world  ', '  test  ']
        })
        cleaner.clean_strings()
        assert cleaner.df['text'].iloc[0] == 'hello'
        assert cleaner.df['text'].iloc[1] == 'world'

    def test_remove_extra_spaces(self):
        """Test removing extra spaces."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'text': ['hello   world', 'foo  bar  baz']
        })
        cleaner.clean_strings()
        assert cleaner.df['text'].iloc[0] == 'hello world'
        assert cleaner.df['text'].iloc[1] == 'foo bar baz'

    def test_standardize_nulls(self):
        """Test standardizing null representations."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'text': ['null', 'NA', 'N/A', '', '-', '.', 'valid']
        })
        cleaner.clean_strings()
        # These should be converted to NaN
        assert pd.isna(cleaner.df['text'].iloc[0])
        assert pd.isna(cleaner.df['text'].iloc[1])
        assert cleaner.df['text'].iloc[6] == 'valid'

    def test_mixed_columns(self):
        """Test cleaning mixed string columns."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': ['  hello  ', 'world', None],
            'col2': [1, 2, 3]
        })
        cleaner.clean_strings()
        assert cleaner.df['col1'].iloc[0] == 'hello'
        assert cleaner.df['col1'].iloc[1] == 'world'


class TestDataTypeConversion:
    """Test data type conversion functionality."""

    def test_infer_objects(self):
        """Test automatic type inference."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'int_col': ['1', '2', '3'],
            'float_col': ['1.5', '2.5', '3.5']
        })
        cleaner.convert_dtypes()
        # Types should be inferred

    def test_no_data_error(self):
        """Test error when no data loaded."""
        cleaner = DataCleaner()
        with pytest.raises(ValueError, match="No data loaded"):
            cleaner.convert_dtypes()


class TestColumnOperations:
    """Test column operations functionality."""

    def test_rename_lowercase(self):
        """Test renaming columns to lowercase."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'ColumnName': [1, 2, 3],
            'Another Column': [4, 5, 6]
        })
        cleaner.rename_columns()
        assert 'columnname' in cleaner.df.columns
        assert 'another_column' in cleaner.df.columns

    def test_rename_special_chars(self):
        """Test renaming columns with special characters."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col-1': [1, 2],
            'col@2': [3, 4],
            'col 3': [5, 6]
        })
        cleaner.rename_columns()
        # Special chars should be removed
        assert 'col1' in cleaner.df.columns
        assert 'col2' in cleaner.df.columns
        assert 'col3' in cleaner.df.columns

    def test_drop_constant_columns(self):
        """Test dropping constant columns."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'const_col': [1, 1, 1, 1],
            'var_col': [1, 2, 3, 4]
        })
        cleaner.drop_constant_columns(threshold=0.99)
        assert 'const_col' not in cleaner.df.columns
        assert 'var_col' in cleaner.df.columns

    def test_drop_low_variance_columns(self):
        """Test dropping low variance columns."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'almost_const': [1, 1, 1, 1, 2],  # Only 2 unique out of 5
            'var_col': [1, 2, 3, 4, 5]
        })
        cleaner.drop_constant_columns(threshold=0.5)
        assert 'almost_const' not in cleaner.df.columns


class TestFullCleaningPipeline:
    """Test the full cleaning pipeline."""

    def test_full_clean(self):
        """Test full cleaning pipeline."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'Col-1': [1.0, 2.0, np.nan, 4.0, 100.0],
            'Col 2': ['  hello  ', 'world', 'NA', 'test', 'data'],
            'const': [1, 1, 1, 1, 1]
        })
        cleaner.full_clean()

        # Check missing values handled
        assert cleaner.df.isnull().sum().sum() == 0

        # Check columns renamed
        assert 'col_1' in cleaner.df.columns
        assert 'col_2' in cleaner.df.columns

        # Check constant column dropped
        assert 'const' not in cleaner.df.columns

    def test_full_clean_with_outliers(self):
        """Test full cleaning pipeline with outlier handling."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 1000],
            'text': ['a', 'b', 'c', 'd', 'e', 'f']
        })
        cleaner.full_clean(handle_outliers=True, outlier_multiplier=1.5)
        # Outlier should be capped
        assert cleaner.df['values'].max() < 1000

    def test_full_clean_without_outliers(self):
        """Test full cleaning pipeline without outlier handling."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 1000],
        })
        cleaner.full_clean(handle_outliers=False)
        # Outlier should remain
        assert cleaner.df['values'].max() == 1000


class TestSQLExport:
    """Test SQL export functionality."""

    def test_export_creates_file(self):
        """Test export creates SQL file."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        cleaner.table_name = 'test_table'
        cleaner.cleaning_report = {'steps_applied': ['test step']}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            output_path = f.name

        try:
            result_path = cleaner.export_to_sql(output_path)
            assert os.path.exists(result_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_content(self):
        """Test export creates valid SQL content."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob']
        })
        cleaner.table_name = 'test'
        cleaner.cleaning_report = {'steps_applied': []}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            output_path = f.name

        try:
            cleaner.export_to_sql(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            assert 'CREATE TABLE' in content
            assert 'INSERT INTO' in content
            assert 'test_cleaned' in content
            assert 'Alice' in content
            assert 'Bob' in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_with_nulls(self):
        """Test export handles null values correctly."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.0, np.nan, 30.0]
        })
        cleaner.table_name = 'test'
        cleaner.cleaning_report = {'steps_applied': []}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            output_path = f.name

        try:
            cleaner.export_to_sql(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            assert 'NULL' in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_with_special_chars(self):
        """Test export escapes special characters in strings."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'text': ["It's a test", "Don't stop"]
        })
        cleaner.table_name = 'test'
        cleaner.cleaning_report = {'steps_applied': []}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            output_path = f.name

        try:
            cleaner.export_to_sql(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            # Single quotes should be escaped
            assert "''" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_timestamp_naming(self):
        """Test export generates timestamp-based filename."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({'col': [1, 2, 3]})
        cleaner.table_name = 'mydata'
        cleaner.cleaning_report = {'steps_applied': []}

        output_path = cleaner.export_to_sql()

        assert 'mydata_cleaned_' in output_path
        assert output_path.endswith('.sql')
        os.unlink(output_path)


class TestCleaningReport:
    """Test cleaning report functionality."""

    def test_get_cleaning_report(self):
        """Test getting cleaning report."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', None]
        })
        cleaner.cleaning_report = {
            'input_file': 'test.sql',
            'initial_rows': 3,
            'initial_columns': 2,
            'steps_applied': ['step1']
        }

        report = cleaner.get_cleaning_report()

        assert report['input_file'] == 'test.sql'
        assert report['final_rows'] == 3
        assert report['final_columns'] == 2
        assert 'steps_applied' in report

    def test_report_tracks_row_changes(self):
        """Test report tracks row count changes."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col': [1, 1, 2, 3]
        })
        cleaner.cleaning_report = {
            'initial_rows': 4,
            'initial_columns': 1,
            'steps_applied': []
        }
        cleaner.handle_duplicates()
        report = cleaner.get_cleaning_report()

        assert report['initial_rows'] == 4
        assert report['final_rows'] == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling empty dataframe."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame()
        cleaner.cleaning_report = {'steps_applied': []}

        # Should not crash
        cleaner.drop_constant_columns()
        assert len(cleaner.df) == 0

    def test_single_row(self):
        """Test handling single row."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'col1': [1],
            'col2': ['test']
        })
        cleaner.cleaning_report = {'steps_applied': []}

        cleaner.full_clean()
        assert len(cleaner.df) == 1

    def test_all_null_column(self):
        """Test handling all-null column."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'valid': [1, 2, 3],
            'all_null': [None, None, None]
        })
        cleaner.cleaning_report = {'steps_applied': []}

        cleaner.handle_missing_values(strategy='auto')
        # Should not crash

    def test_large_dataset_simulation(self):
        """Test handling larger dataset."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        cleaner.cleaning_report = {
            'initial_rows': 10000,
            'initial_columns': 3,
            'steps_applied': []
        }

        cleaner.full_clean()
        assert len(cleaner.df) > 0

    def test_datetime_columns(self):
        """Test handling datetime columns."""
        cleaner = DataCleaner()
        cleaner.df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        cleaner.cleaning_report = {'steps_applied': []}

        # Should not crash on datetime columns
        cleaner.rename_columns()
        cleaner.handle_missing_values()


# Integration test helper
def create_test_sql_file(query: str) -> str:
    """Helper to create temporary SQL file for testing."""
    fd, path = tempfile.mkstemp(suffix='.sql')
    with os.fdopen(fd, 'w') as f:
        f.write(query)
    return path


class TestIntegration:
    """Integration tests requiring database connection."""

    def test_full_workflow_with_sqlite(self):
        """Test complete workflow with SQLite in-memory database."""
        import sqlite3

        # Create in-memory database with test data
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE test_data (
                id INTEGER,
                name TEXT,
                value REAL
            )
        ''')
        cursor.executemany(
            'INSERT INTO test_data VALUES (?, ?, ?)',
            [
                (1, 'Alice', 10.5),
                (2, 'Bob', 20.3),
                (3, 'Charlie', 30.1),
                (1, 'Alice', 10.5),  # Duplicate
                (4, None, 40.0),     # Null name
            ]
        )
        conn.commit()

        # Create SQL file
        sql_path = create_test_sql_file("SELECT * FROM test_data")

        try:
            # Test DataCleaner
            cleaner = DataCleaner("sqlite:///:memory:")
            cleaner.load_from_sql_file(sql_path)

            assert len(cleaner.df) == 5
            assert cleaner.df.isnull().sum().sum() == 1

            cleaner.full_clean()

            # Duplicates removed, nulls filled
            assert len(cleaner.df) <= 4
            assert cleaner.df.isnull().sum().sum() == 0

        finally:
            conn.close()
            os.unlink(sql_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
