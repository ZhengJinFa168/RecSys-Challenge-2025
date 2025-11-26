import pandas as pd
import argparse
import sys
from pathlib import Path


def compare_csv_files(file1, file2, key_columns=None, ignore_columns=None):
    """
    Compare two CSV files and return differences

    Args:
        file1 (str): Path to first CSV file
        file2 (str): Path to second CSV file
        key_columns (list): List of column names to use as keys for comparison
        ignore_columns (list): List of column names to ignore in comparison

    Returns:
        dict: Dictionary containing comparison results
    """

    # Read CSV files
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        return {"error": f"Error reading files: {e}"}

    # Basic info
    result = {
        "files_identical": False,
        "file1_rows": len(df1),
        "file2_rows": len(df2),
        "file1_columns": list(df1.columns),
        "file2_columns": list(df2.columns),
        "differences": {}
    }

    # Check if column names match
    if set(df1.columns) != set(df2.columns):
        result["differences"]["columns"] = {
            "missing_in_file2": list(set(df1.columns) - set(df2.columns)),
            "missing_in_file1": list(set(df2.columns) - set(df1.columns))
        }
        return result

    # Reorder columns to match if they're in different order
    df2 = df2[df1.columns]

    # Handle ignore columns
    if ignore_columns:
        df1 = df1.drop(columns=ignore_columns, errors='ignore')
        df2 = df2.drop(columns=ignore_columns, errors='ignore')

    # Determine key columns for comparison
    if key_columns is None:
        # If no key columns specified, use all columns
        key_columns = list(df1.columns)

    # Check for missing key columns
    missing_keys = [col for col in key_columns if col not in df1.columns]
    if missing_keys:
        return {"error": f"Key columns not found: {missing_keys}"}

    # Add temporary index for tracking
    df1['_temp_index'] = range(len(df1))
    df2['_temp_index'] = range(len(df2))

    # Create combined key for comparison
    if len(key_columns) == 1:
        df1['_combined_key'] = df1[key_columns[0]].astype(str)
        df2['_combined_key'] = df2[key_columns[0]].astype(str)
    else:
        df1['_combined_key'] = df1[key_columns].astype(str).agg('|'.join, axis=1)
        df2['_combined_key'] = df2[key_columns].astype(str).agg('|'.join, axis=1)

    # Find differences
    all_keys = set(df1['_combined_key']) | set(df2['_combined_key'])

    missing_in_file2 = []
    missing_in_file1 = []
    different_rows = []

    for key in all_keys:
        row1 = df1[df1['_combined_key'] == key]
        row2 = df2[df2['_combined_key'] == key]

        if row1.empty and not row2.empty:
            missing_in_file1.append({
                'key': key,
                'row_index_file2': int(row2['_temp_index'].iloc[0]),
                'data': row2.drop(columns=['_temp_index', '_combined_key']).iloc[0].to_dict()
            })
        elif not row1.empty and row2.empty:
            missing_in_file2.append({
                'key': key,
                'row_index_file1': int(row1['_temp_index'].iloc[0]),
                'data': row1.drop(columns=['_temp_index', '_combined_key']).iloc[0].to_dict()
            })
        elif not row1.empty and not row2.empty:
            # Compare all columns except temporary ones
            compare_cols = [col for col in df1.columns if col not in ['_temp_index', '_combined_key']]
            diff_cols = []

            for col in compare_cols:
                val1 = row1[col].iloc[0]
                val2 = row2[col].iloc[0]

                # Handle NaN values
                if pd.isna(val1) and pd.isna(val2):
                    continue
                elif pd.isna(val1) or pd.isna(val2) or val1 != val2:
                    diff_cols.append({
                        'column': col,
                        'file1_value': val1,
                        'file2_value': val2
                    })

            if diff_cols:
                different_rows.append({
                    'key': key,
                    'row_index_file1': int(row1['_temp_index'].iloc[0]),
                    'row_index_file2': int(row2['_temp_index'].iloc[0]),
                    'differences': diff_cols
                })

    result["differences"]["missing_in_file2"] = missing_in_file2
    result["differences"]["missing_in_file1"] = missing_in_file1
    result["differences"]["different_values"] = different_rows

    # Check if files are identical
    result["files_identical"] = (
            len(missing_in_file2) == 0 and
            len(missing_in_file1) == 0 and
            len(different_rows) == 0 and
            len(df1) == len(df2)
    )

    return result


def print_results(results):
    """Print comparison results in a readable format"""

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return

    print(f"📊 Comparison Results:")
    print(f"File 1: {results['file1_rows']} rows, {len(results['file1_columns'])} columns")
    print(f"File 2: {results['file2_rows']} rows, {len(results['file2_columns'])} columns")
    print()

    if results["files_identical"]:
        print("✅ The files are identical!")
        return

    # Print column differences
    if "columns" in results["differences"]:
        print("🔍 Column Differences:")
        if results["differences"]["columns"]["missing_in_file2"]:
            print(f"  Columns in File 1 but not in File 2: {results['differences']['columns']['missing_in_file2']}")
        if results["differences"]["columns"]["missing_in_file1"]:
            print(f"  Columns in File 2 but not in File 1: {results['differences']['columns']['missing_in_file1']}")
        print()

    # Print row differences
    diff = results["differences"]

    if diff["missing_in_file2"]:
        print(f"❌ Rows in File 1 but not in File 2: {len(diff['missing_in_file2'])}")
        for i, row in enumerate(diff["missing_in_file2"][:5]):  # Show first 5
            print(f"  Row {row['row_index_file1'] + 1} (Key: {row['key']})")
        if len(diff["missing_in_file2"]) > 5:
            print(f"  ... and {len(diff['missing_in_file2']) - 5} more")
        print()

    if diff["missing_in_file1"]:
        print(f"❌ Rows in File 2 but not in File 1: {len(diff['missing_in_file1'])}")
        for i, row in enumerate(diff["missing_in_file1"][:5]):
            print(f"  Row {row['row_index_file2'] + 1} (Key: {row['key']})")
        if len(diff["missing_in_file1"]) > 5:
            print(f"  ... and {len(diff['missing_in_file1']) - 5} more")
        print()

    if diff["different_values"]:
        print(f"🔄 Rows with different values: {len(diff['different_values'])}")
        for i, row in enumerate(diff["different_values"][:3]):  # Show first 3
            print(f"  Row File1:{row['row_index_file1'] + 1} File2:{row['row_index_file2'] + 1} (Key: {row['key']})")
            for change in row["differences"][:3]:  # Show first 3 changes per row
                print(f"    {change['column']}: '{change['file1_value']}' → '{change['file2_value']}'")
            if len(row["differences"]) > 3:
                print(f"    ... and {len(row['differences']) - 3} more changes")
            print()
        if len(diff["different_values"]) > 3:
            print(f"  ... and {len(diff['different_values']) - 3} more rows with differences")


def main():
    parser = argparse.ArgumentParser(description='Compare two CSV files')
    parser.add_argument('file1', help='First CSV file')
    parser.add_argument('file2', help='Second CSV file')
    parser.add_argument('--key-columns', nargs='+', help='Column names to use as keys for comparison')
    parser.add_argument('--ignore-columns', nargs='+', help='Column names to ignore in comparison')
    parser.add_argument('--output', choices=['summary', 'detailed'], default='summary',
                        help='Output detail level (summary/detailed)')

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.file1).exists():
        print(f"Error: File '{args.file1}' not found")
        sys.exit(1)
    if not Path(args.file2).exists():
        print(f"Error: File '{args.file2}' not found")
        sys.exit(1)

    print(f"Comparing '{args.file1}' with '{args.file2}'...")
    print()

    results = compare_csv_files(args.file1, args.file2, args.key_columns, args.ignore_columns)
    print_results(results)


if __name__ == "__main__":
    results = compare_csv_files("output.csv", "output2.csv")
    print_results(results)