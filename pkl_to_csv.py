#!/usr/bin/env python3
"""
Convert a pickle file to CSV.
Usage:
    python pkl_to_csv.py input.pkl output.csv
"""
import argparse
import pickle
import pandas as pd
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert a pickle file to a CSV file")
    parser.add_argument("input", help="Path to the input .pkl file")
    parser.add_argument("output", help="Path to the output .csv file")
    args = parser.parse_args()

    try:
        with open(args.input, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}", file=sys.stderr)
        sys.exit(1)

    # If it's already a DataFrame, use it; otherwise try to build one
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            print(f"Error converting data to DataFrame: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        df.to_csv(args.output, index=False)
    except Exception as e:
        print(f"Error writing CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully wrote CSV to {args.output}")


if __name__ == "__main__":
    main()
