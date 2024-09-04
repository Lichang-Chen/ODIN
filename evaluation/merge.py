import argparse
import json
import os

def merge_json_files(file1, file2, files_path, output_path):
    merged_data = []
    with open(os.path.join(files_path, file1), 'r') as f1:
        data1 = json.load(f1)
    
    with open(os.path.join(files_path, file2), 'r') as f2:
        data2 = json.load(f2)
    
    entry1 = file1.split(".json")[0]
    entry2 = file2.split(".json")[0]
    for i in range(len(data1)):
        merged_data.append(
            {
                "id": i,
                "prompt": data1[i]['prompt'],
                f"{entry1}": data1[i]['answer'],
                f"{entry2}": data2[i]['answer'] 
            }
        )
    with open(output_path, 'w') as outfile:
        json.dump(merged_data, outfile, indent=2)
    
    print("JSON files merged successfully!")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument(
        "--files_path",
        type=str, default="generation", 
        help="the path to the generation files"
    )
    parser.add_argument(
        "-f1", "--file1",
        type=str, help="the first file to merge"
    )
    parser.add_argument(
        "-f2", "--file2",
        type=str, help="the second file to merge"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The output path."
    )
    args = parser.parse_args()
    merge_json_files(args.file1, args.file2, args.files_path, args.output_path)