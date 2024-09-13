import csv
from datetime import datetime, timedelta

def fill_dates(input_file, output_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Get the starting date from row 2
    start_date = datetime.strptime(data[1][0], '%m/%d/%y')

    # Fill in the dates
    for i in range(1, len(data)):
        data[i][0] = (start_date - timedelta(days=i-1)).strftime('%m/%d/%y')

    # Write the updated data to the output file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Use the function
fill_dates('lens.csv', 'lens_with_dates.csv')

print("Dates have been filled and saved to lens_with_dates.csv")