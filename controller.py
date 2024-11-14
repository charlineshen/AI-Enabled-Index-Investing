import argparse
import os
import glob
from cli import chunk, embed, load, chunk_with_arguments, chat_with_arguments, INPUT_FOLDER, OUTPUT_FOLDER
from data_process import process_input_pdfs

'''
User need to specify: input folder with all documents, question list csv file.

Design Choices:
    a. command line for user to specify input aand output file path
    b. json file for user to specify input and output file path, configurations

1. call data_process
    - input: folder with all documents
    - add function in data_process.py, process all pdfs in the folder to txts
    
2. skip chunk and embed for files already in chromadb
    - for each pdf file name, run check function
    - if not in chromadb, run chunk, embed, and load

3. call chat
    - input: question list csv file, pdf names
    - return 
        - an output comparasion csv

'''

OUTPUT_TABLE_FOLDER = "output_tables"

if __name__ == "__main__":
    # The first user input is the folder name containing all the pdfs, the second user input is the question file.
    # Extract the argument
    parser = argparse.ArgumentParser(description='Generate index comparison table')
    parser.add_argument('input_folder_name', type=str, help='Input folder containing PDFs')
    parser.add_argument('question_file', type=str, help='CSV file containing questions')
    args = parser.parse_args()
    input_folder_name = args.input_folder_name
    question_file = args.question_file

    input_folder_path = INPUT_FOLDER + '/' + input_folder_name
    output_file_path = OUTPUT_TABLE_FOLDER + '/' + input_folder_name + '.csv'

    # TODO check not too many files in the input_folder_path
    
    # call data_process
    process_input_pdfs(input_folder_name)

    # build RAG knowledge database for the input_folder
    chunk_with_arguments(input_folder_name)
    embed()
    load()

    # Get the list of embedding files
    jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"embeddings-semantic-split-*.jsonl"))
    jsonl_files.extend(glob.glob(os.path.join(OUTPUT_FOLDER, f"chunks-semantic-split-*.jsonl")))
    print("Number of files to remove:", len(jsonl_files))
    # Remove files in output folder
    for file in jsonl_files:
        try:
            os.remove(file)
            print(f"COMPLETE knowlege retrival for {file}. Now Remove the file...")
        except Exception as e:
            print(f"COMPLETE knowlege retrival for {file}. Error in removing the file... {file}: {e}")

    # call chat
    # NOTE output as csv?
    chat_with_arguments(input_folder_name, question_file, output_file_path)
