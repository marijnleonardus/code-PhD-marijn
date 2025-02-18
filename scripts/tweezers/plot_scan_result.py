# author: Marijn Venderbosch
# January 2025

import numpy as np
import csv
import os

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from data_handling_class import compute_avg_std

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'T:\\KAT1\\Marijn\\tweezers\\scan magn field\\'


# Main block to use the function
def main():
    # Compute results
    filename = "bz0"
    results = compute_avg_std(images_path + filename + ".csv")

    # Print the results
    print("x, y, yerr")
    for x, avg, std in results:
        print(f"{x}, {avg:.4f}, {std:.4f}")

    # write the results to a new CSV file
    output_csv = images_path +  filename + "processed.csv"
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "yerr"])
        writer.writerows(results)

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()
