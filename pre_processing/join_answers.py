import pandas as pd
import sys
import argparse

# Read the two CSV files
parser = argparse.ArgumentParser(description='Join two CSV files')
parser.add_argument('--file1', default='data/processed/answers_human.csv', help='Path to the first CSV file')
parser.add_argument('--file2', default='data/processed/answers_vlms.csv', help='Path to the second CSV file')
parser.add_argument('--output', default='data/r2.csv', help='Path to the output CSV file')
args = parser.parse_args()

df1 = pd.read_csv(args.file1)
df2 = pd.read_csv(args.file2)

# Concatenate the dataframes
result = pd.concat([df1, df2], ignore_index=True)
# Save to a new CSV file
result.to_csv(args.output, index=False)

print("CSV files joined successfully!")