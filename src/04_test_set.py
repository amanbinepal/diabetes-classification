# 04_split_test_features.py
# Author: Vy Phan
# Date: 2025-12-11

# This script splits the test dataset into features (X_test) and target (y_test),
# and saves them as separate CSV files in a user-specified directory.

# Usage:
# python src/04_split_test_features.py --test-file=src/objects/test_df.csv --output-dir=src/objects

import click
import pandas as pd

@click.command()
@click.option('--test-file', default='src/objects/test_df.csv', help='Test CSV file')

def main(test_file):
    """
    Split test data into features (X_test) and target (y_test) and save as CSV files.

    Parameters
    ----------
    test_file : str
        Path to the CSV file containing the test data.
    output_dir : str
        Folder where X_test.csv and y_test.csv will be saved.
    """

    test_df = pd.read_csv(test_file)

    X_test, y_test = (
        test_df.drop(columns=["Diabetes_binary"]),
        test_df["Diabetes_binary"],
    )

    # Save as CSV
    X_test.to_csv('src/objects/X_test.csv', index=False)
    y_test.to_csv('src/objects/y_test.csv', index=False)

    click.echo("Ran test set")

if __name__ == '__main__':
    main()
