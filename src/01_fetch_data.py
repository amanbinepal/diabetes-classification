# 01_fetch_data.py
# Author: Vy Phan
# Date: 2025-12-11

# This script downloads the CDC Diabetes dataset from the UC Irvine Machine Learning Repository
# using the ucimlrepo package, and saves it as a CSV file in the data/raw directory.

# Usage: python src/01_fetch_data.py

import click
from pathlib import Path
from ucimlrepo import fetch_ucirepo

@click.command()

def main():
    """
    Download CDC Diabetes dataset and save it as a CSV in data/raw directory.
    
    The dataset ID used for UCIMLRepo is 891 (Diabetes Binary Health Indicators).
    """
    
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    dat = cdc_diabetes_health_indicators.data.original
    
    output_path = 'data/raw/diabetes_binary_health_indicators_BRFSS2015.csv'
    dat.to_csv(output_path, index=False)

    click.echo("Ran fetch")
    
if __name__ == '__main__':
    main()