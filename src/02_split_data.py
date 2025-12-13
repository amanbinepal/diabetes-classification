# 02_split_data.py
# Author: Vy Phan
# Date: 2025-12-11

# This script splits the raw CDC Diabetes dataset into training and testing sets (80/20 split)
# and saves the resulting datasets as CSV files in the src/objects directory.

# Usage: python src/02_split_data.py --input-file=data/raw/diabetes_binary_health_indicators_BRFSS2015.csv

import click
import pandas as pd
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--input-file', default='data/raw/diabetes_binary_health_indicators_BRFSS2015.csv', help='Input CSV file')

def main(input_file):
    """
    Split raw CDC Diabetes dataset into training and testing sets and save as CSV.

    Parameters
    ----------
    input_file : str
        Path to the raw CSV file to be split.
    """

    dat = pd.read_csv(input_file)
    train_df, test_df = train_test_split(dat, test_size=0.2, random_state=522)

    # Save as CSV
    train_df.to_csv('src/objects/train_df.csv', index=False)
    test_df.to_csv('src/objects/test_df.csv', index=False)

    click.echo(
        "-------------Ran split-------------\nSaved:\nsrc/objects/train_df.csv\nsrc/objects/test_df.csv\n"
    )

if __name__ == '__main__':
    main()
