import csv
import random

# Read the data from the CSV file
with open('monkey.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Separate the header row from the data
header = data[0]
data = data[1:]

# Shuffle the data by sampling randomly
random.shuffle(data)

# Insert the header back at the beginning
data.insert(0, header)

# Save the randomized data to a new CSV file
with open('randomized_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
