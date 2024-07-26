import os
import json


def generate_jsonl_from_folder(folder_path, output_file):
    # Create an empty list to store JSON objects
    jsonl_data = []

    # Traverse files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is an image file
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Generate corresponding description file name
            base_name, _ = os.path.splitext(file_name)
            description_file = base_name + '.txt'
            description_path = os.path.join(folder_path, description_file)

            # Read the description file content
            if os.path.exists(description_path):
                with open(description_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()  # Read and remove leading and trailing whitespace

                # Construct JSON object
                json_obj = {
                    "file_name": file_name,
                    "text": text
                }
                jsonl_data.append(json_obj)

    # Write data to jsonl file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# Example usage
folder_path = '../cartoon/cartoon_dataset'  # Folder containing images and description files
output_file = folder_path + '/metadata.jsonl'  # Generated jsonl file name
print(output_file)
generate_jsonl_from_folder(folder_path, output_file)
