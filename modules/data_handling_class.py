import pandas as pd


def pandas_read_datfile(file_path, file_name):
    try:
        data = pd.read_csv(file_path + "\\" + file_name, comment='#', sep='\s+')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Check if the file exists at: {data_location}")
    return data


def main():
    file_path = r"\\physstor\cqt-t\KAT1\Marijn\FC1500measurements\cavity_drift"
    file_name = r"august9result.dat"
    dataset = pandas_read_datfile(file_path, file_name)


if __name__ == "__main__":
    main()
