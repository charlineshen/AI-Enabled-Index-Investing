import argparse
import os
import glob
from cli import embed, load, chunk, chat, INPUT_FOLDER, OUTPUT_FOLDER, QUESTION_FOLDER
from data_process import process_input_pdfs, check_num_input_pdfs

OUTPUT_TABLE_FOLDER = "output_tables"

if __name__ == "__main__":
    # The first user input is the folder name containing all the pdfs, the second user input is the question file.
    # Extract the argument
    parser = argparse.ArgumentParser(description='Generate index comparison table')
    parser.add_argument('input_folder_name', type=str, help='Input folder containing PDFs')
    parser.add_argument('question_file_name', type=str, help='CSV file containing questions')
    args = parser.parse_args()
    input_folder_name = args.input_folder_name
    question_file_name = args.question_file_name

    input_folder_path = INPUT_FOLDER + '/' + input_folder_name
    output_file_path = OUTPUT_TABLE_FOLDER + '/' + input_folder_name + '.xlsx'

    # Make sure users won't accidentally index too many files in input_folder_path
    if not check_num_input_pdfs(input_folder_name, threshold = 20):
        exit(1)

    # Call data_process
    print('=============STEP 1. Initial processing of input pdfs=============')
    process_input_pdfs(input_folder_name)

    # Build RAG knowledge database for the input_folder
    print('\n=============STEP 2. Dividing input texts into chunks=============')
    chunk(input_folder_name)
    print('\n=====================STEP 3. Embedding chunks=====================')
    embed()
    print('\n===========STEP 4. Loading embedded chunks into database==========')
    load()

    # Get the list of embedding files
    jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"embeddings-semantic-split-*.jsonl"))
    jsonl_files.extend(glob.glob(os.path.join(OUTPUT_FOLDER, f"chunks-semantic-split-*.jsonl")))

    # Remove files in output folder
    for file in jsonl_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"COMPLETE knowlege retrival for {file}. Error in removing the file... {file}: {e}")

    # Call chat
    print('\n======================STEP 5. Initiating chat=====================')
    chat(input_folder_name, question_file_name, output_file_path)
