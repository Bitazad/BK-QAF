import json
import sys

input_file_path = sys.argv[1]

# Load the JSON file
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Filter questions with type "factoid"
factoid_questions = [question for question in data['questions'] if question['type'] == 'factoid']

# Create a new dictionary with filtered questions
filtered_data = {'questions': factoid_questions}

# Save the filtered data to a new JSON file
output_file_path = sys.argv[2]
with open(output_file_path, 'w') as output_file:
    json.dump(filtered_data, output_file, indent=2)

print(f"Factoid questions have been saved to {output_file_path}.")
