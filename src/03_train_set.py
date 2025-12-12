# 03_split_train_features.py
# Author: Vy Phan
# Date: 2025-12-11

# This script splits the training dataset into features (X_train) and target (y_train),
# and saves them as separate CSV files in a specified directory.

# Usage: python src/03_split_train_features.py --train-file=src/objects/train_df.csv --output-dir=src/objects

import click
import pandas as pd

@click.command()
@click.option('--train-file', default='src/objects/train_df.csv', help='Training dataset CSV file')

def main(train_file):
    """
    Split training data into features (X_train) and target (y_train) and save as CSV files.

    Parameters
    ----------
    train_file : str
        Path to the CSV file containing the training data.
    output_dir : str
        Folder where X_train.csv and y_train.csv will be saved.
    """

    train_df = pd.read_csv(train_file)

    X_train, y_train = (
        train_df.drop(columns=["Diabetes_binary"]),
        train_df["Diabetes_binary"],
    )

    # Save as CSV
    X_train.to_csv('src/objects/X_train.csv', index=False)
    y_train.to_csv('src/objects/y_train.csv', index=False)

    click.echo("Ran train set")

if __name__ == '__main__':
    main()
