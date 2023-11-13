import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', '-t', type=str, help='rouge-1, rouge-2, or rouge-l')
args = parser.parse_args()

x = list()
y = list()
for i in range(10):
    input_file = f'result_e{i}'
    with open(input_file, 'r', encoding='utf-8') as json_file:
        my_dict = json.load(json_file)
    x.append(i + 1)
    y.append(my_dict[args.target]["f"] * 100)


# Step 3: Create Data
plt.xlabel('epoch')
plt.ylabel(args.target)

# Step 4: Plot the Data
plt.plot(x, y)
plt.xticks(x)
plt.scatter(x, y, color='black', marker='.', label='Data Points')
if args.target == 'rouge-1':
    plt.plot([1, 10], [22, 22], color='blue', linestyle='--', label='Straight Line')
if args.target == 'rouge-2':
    plt.plot([1, 10], [8.5, 8.5], color='blue', linestyle='--', label='Straight Line')
if args.target == 'rouge-l':
    plt.plot([1, 10], [20.5, 20.5], color='blue', linestyle='--', label='Straight Line')
# Step 5: Show the Chart
plt.show()

