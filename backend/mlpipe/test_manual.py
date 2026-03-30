"""
Manual test script for general_data_cleaning module.

Usage:
    python test_manual.py
"""

from general_data_cleaning import DataCleaner


def main():
    """Run manual test with the SQL file (no database connection needed)."""
    # SQL file path
    sql_file = r"Cases Reported, Persons Injured and Died_Sample_Data.sql"

    # Initialize cleaner (no connection string needed - uses in-memory SQLite)
    cleaner = DataCleaner()

    # Load data from SQL file (uses in-memory SQLite database)
    print(f"Loading data from: {sql_file}")
    cleaner.load_from_sql_file(sql_file, use_in_memory_db=True)

    # Apply full cleaning pipeline (includes outlier handling)
    print("Applying cleaning operations...")
    cleaner.full_clean(handle_outliers=True)

    # Export cleaned data to SQL file in same folder
    output_path = cleaner.export_to_sql()

    # Print summary
    report = cleaner.get_cleaning_report()
    print(f"\nCleaning complete!")
    print(f"Rows: {report['initial_rows']} -> {report['final_rows']}")
    print(f"Columns: {report['initial_columns']} -> {report['final_columns']}")
    print(f"Output: {output_path}")


# if __name__ == "__main__":
main()
