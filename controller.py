import argparse
import os
from cli import chunk, embed, load, chat_with_arguments, check_duplicates
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


if __name__ == "__main__":
    # the first user input is the folder name containing all the pdfs, the second user input is the question file.
    # Extract the argument
    parser = argparse.ArgumentParser(description='Generate index comparison table')
    parser.add_argument('input_folder_name', type=str, help='Input folder containing PDFs')
    parser.add_argument('question_file', type=str, help='CSV file containing questions')
    args = parser.parse_args()
    input_folder_name = args.input_folder_name
    question_file = args.question_file

    input_folder_path = 'inputs' + input_folder_name
    output_file_path = 'output_tables/' + input_folder_name + '.csv'

    # call data_process
    process_input_pdfs(input_folder_name)

    # for each pdf file in the folder, call chunk, embed, and load; skip if already in chromadb
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.pdf'):
            title_name = file_name.split('.')[0]
            if not check_duplicates(title_name):
                chunk()
                embed()
                load()

    # call chat
    chat_with_arguments(input_folder_name, question_file, output_file_path)