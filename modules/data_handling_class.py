import pandas as pd
import csv
from collections import defaultdict
import numpy as np


def pandas_read_datfile(file_path, file_name):
    try:
        data = pd.read_csv(file_path + "\\" + file_name, comment='#', sep='\s+')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Check if the file exists at: {data_location}")
    return data


def read_csv_file(file_path, file_name):
    # Combine the path and filename
    file_name_full = file_path + "\\" + file_name
    
    try:
        # Open the CSV file
        with open(file_name_full, newline='') as csvfile:
            # Read the CSV file
            csv_reader = csv.reader(csvfile, delimiter=',')

            # Skip the header (first line)
            next(csv_reader)
            
            # Initialize a list to store the rows
            data = []
            
            # Iterate over each row in the csv_reader and append to data list
            for row in csv_reader:
                data.append(row)
            
        # Return the list of rows
        return data
    
    except FileNotFoundError:
        print(f"Error: The file {file_name_full} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def compute_avg_std(csv_file):
    """given a csv file, with x data in column 1 and y data in column 2,
    where x is repeated multiple times, this function
    computes the avg and std dev in the corresopnding y values
    
    generated by chatGPT"""
    # Dictionary to store lists of y values for each x
    data = defaultdict(list)

    # Read the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                x = float(row[0])
                y = float(row[1])
                data[x].append(y)
            except ValueError:
                # Skip rows with invalid data
                continue

    # Compute averages and standard deviations
    results = []
    for x, y_values in data.items():
        avg = np.mean(y_values)
        std = np.std(y_values)
        results.append((x, avg, std))

    # Sort results by x for better readability
    results.sort(key=lambda entry: entry[0])

    return results


def main():
    file_path = r"\\physstor\cqt-t\KAT1\Comb_measurements"
    file_name = r"AU06792-log_reprate_beat.csv"
    dataset = read_csv_file(file_path, file_name)

    # print first 3 lines to check it went correctly
    for row in dataset[:3]:  
        print(row)

if __name__ == "__main__":
    main()
