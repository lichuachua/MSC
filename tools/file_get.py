import os
import json


def get_dataset_files(dataset_folder):
    """ 获取指定文件夹中的所有文件名 """
    return {f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))}


def filter_metadata(metadata_file, dataset_files, output_file):
    """ 从 metadata.jsonl 中筛选出文件夹中存在的文件名的 JSON 行，并写入到输出文件 """
    with open(metadata_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            file_name = data.get('file_name')
            if file_name in dataset_files:
                outfile.write(json.dumps(data) + '\n')


def main():
    # 设置文件夹路径和 metadata.jsonl 文件路径
    dataset_folder = '/Users/lichuachua/Downloads/MSC/lora/cartoon_new/cartoon_dataset'
    metadata_file = '/Users/lichuachua/Downloads/MSC/lora/cartoon_new/cartoon_dataset/metadata.jsonl'
    output_file = '/Users/lichuachua/Downloads/MSC/lora/cartoon_new/cartoon_dataset/metadata.jsonl'

    # 获取文件夹中的所有文件名
    dataset_files = get_dataset_files(dataset_folder)

    # 筛选 metadata 并写入输出文件
    filter_metadata(metadata_file, dataset_files, output_file)

    print(f"Filtered metadata saved to {output_file}")


if __name__ == '__main__':
    main()
