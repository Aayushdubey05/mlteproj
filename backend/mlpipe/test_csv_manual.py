"""
Manual test script for general_csv_data_cleaning module.

Usage:
    python test_csv_manual.py
"""

from general_csv_data_cleaning import CsvDataCleaner


def main():
    """Run manual test with the CSV file."""
    # CSV file path (update this to the actual CSV file path)
    csv_file = r"Cases Reported, Persons Injured and Died_Sample_Data.csv"

    # Initialize cleaner
    cleaner = CsvDataCleaner()

    # Load data from CSV file
    print(f"Loading data from: {csv_file}")
    cleaner.load_from_csv_file(csv_file)

    # Apply full cleaning pipeline (includes outlier handling)
    print("Applying cleaning operations...")
    cleaner.full_clean(handle_outliers=True)

    # Export cleaned data to CSV file in same folder
    output_path = cleaner.export_to_csv()

    # Print summary
    report = cleaner.get_cleaning_report()
    print(f"\nCleaning complete!")
    print(f"Rows: {report['initial_rows']} -> {report['final_rows']}")
    print(f"Columns: {report['initial_columns']} -> {report['final_columns']}")
    print(f"Output: {output_path}")


# if __name__ == "__main__":
main()