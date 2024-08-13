import os
import json
import csv


def generate_files_from_folder(folder_path, output_jsonl_file, output_csv_file):
    # Create an empty list to store JSON objects and CSV rows
    jsonl_data = []
    csv_rows = []

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

                # Construct CSV row
                csv_row = {
                    "file_name": file_name,
                    "text": text
                }
                csv_rows.append(csv_row)

    # Write data to JSONL file
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Write data to CSV file
    with open(output_csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(csv_rows)


# Example usage
folder_path = '../cartoon/cartoon_dataset'  # Folder containing images and description files
output_jsonl_file = os.path.join(folder_path, 'metadata.jsonl')  # Generated JSONL file name
output_csv_file = os.path.join(folder_path, 'metadata.csv')  # Generated CSV file name
print(f"JSONL output file: {output_jsonl_file}")
print(f"CSV output file: {output_csv_file}")
generate_files_from_folder(folder_path, output_jsonl_file, output_csv_file)
