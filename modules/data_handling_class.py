import pandas as pd
import csv


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


def main():
    file_path = r"\\physstor\cqt-t\KAT1\Comb_measurements"
    file_name = r"AU06792-log_reprate_beat.csv"
    dataset = read_csv_file(file_path, file_name)

    # print first 3 lines to check it went correctly
    for row in dataset[:3]:  
        print(row)

if __name__ == "__main__":
    main()
