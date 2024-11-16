import PyPDF2
import os
import re
import unidecode

# Find the line index of header and footer
def find_common_index(reader):
    page_line_lengths = []
        
    # Only check the first 5 pages (skip cover) to find common lines
    for page_num in range(1, min(6, len(reader.pages))):
        page = reader.pages[page_num]
        text = page.extract_text()
        lines = text.splitlines()
        page_line_lengths.append([len(line) for line in lines])
            
    for i in range(len(page_line_lengths[0])):
        if not all(lst[i] == page_line_lengths[0][i] for lst in page_line_lengths):
            # Return the default common line index of footer and header, as well as the list of line lengths
            return i, page_line_lengths[0][:i]
    
    return 0, []

# Return a list of strings; each string is a page's content without header and footer (cover removed)
def extract_text_list(reader, common_index, common_lengths):
    full_text = []
    
    for page_num in range(1, len(reader.pages)):
        # Read in the text by lines
        page = reader.pages[page_num]
        text = page.extract_text()
        lines = text.splitlines()
        
        # Remove the common lines
        header_footer = lines[:common_index]
        header_footer_lengths = [len(line) for line in header_footer]
        if header_footer_lengths == common_lengths:
            cleaned_lines = lines[common_index:]
        # If the line lengths don't match exactly, the header and footer might have ended before common_index
        else:
            new_common_index = common_index
            for i in range(len(header_footer_lengths)):
                # Only change the index when the line is longer than the reference length by 1 (for page numbers 10-99)
                # or by 2 (for page numbers 100-999), etc., or when the line is shorter than the reference length
                if header_footer_lengths[i] - common_lengths[i] >= len(str(page_num+1)) or \
                    header_footer_lengths[i] - common_lengths[i] < 0:
                    new_common_index = i
                    break
            cleaned_lines = lines[new_common_index:]
        
        # Join the remaining lines into a single string for each page
        page_text = "\n".join(cleaned_lines)
        full_text.append(page_text)
    
    return full_text

# Returns a mega string of the entire PDF without table of contents, contact us, and disclaimer
def preprocess_text(text_list):
    # Search for the contacts page and the disclaimer page
    contacts_pattern = re.compile(r'\+ 1 \d{3} \d{3} \d{4}') # US phone number
    disclaimer_pattern = re.compile(r'This document and all of the information contained in it') # disclaimer-like content
    contacts, disclaimer_page = None, None
    for page, page_text in enumerate(text_list): # Search for the last occurence of contacts/disclaimer pattern
        if contacts_pattern.search(page_text):
            contacts = page_text # Save content instead of index to avoid indexing errors when removing two items
        if disclaimer_pattern.search(page_text):
            disclaimer_page = page
    # Remove the contacts page and the disclaimer page, if found
    if contacts:
        text_list.remove(contacts)
    if (disclaimer_page) and (disclaimer_page >= len(text_list) - 3):
        text_list = text_list[:disclaimer_page]

    # Search for and remove the table of contents
    page_number_pattern = re.compile(r'(\d+)\s*$')  # Digits followed by trailing whitespaces, at end of line
    start_page = 0
    for line in text_list[0].split("\n"): # Search in first page, by lines
        page_number_match = page_number_pattern.search(line) # Search for first occurrence of the pattern
        if page_number_match:
            start_page = int(page_number_match.group(1)) - 2 # -2 due to zero-indexing and cover page
            break
    # Restore if the match is suspiciously large
    if start_page > len(text_list) // 10 + 1:
        start_page = 0

    # Merge list of text into long string
    text = "\n".join(text_list[start_page:])
    return text

def save_text_to_file(text, text_dir, pdf_name):
    # Extract the filename without extension
    base_name = pdf_name.replace('.pdf', '.txt')
    # Construct the new path for the .txt file
    txt_path = os.path.join(text_dir, base_name)
    # Write the extracted text to the file
    os.makedirs(text_dir, exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)
    print(f"Text saved to: {txt_path}")
    return txt_path

# Make sure users won't accidentally index too many files in input_folder_path
def check_num_input_pdfs(folder_name, threshold = 20):
    folder_path = 'input_pdfs/' + folder_name
    pdf_count = sum(1 for file in os.listdir(folder_path) if file.lower().endswith('.pdf'))
    
    if pdf_count >= threshold:
        print(f"Too many PDFs in {folder_path}. {pdf_count} PDFs found.")
        while True:
            user_response = input("Do you want to proceed? (y/n): ").strip().lower()
            if user_response in ['y', 'yes']:
                return True
            elif user_response in ['n', 'no']:
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    return True


# Process the PDFs in a folder to txts
def process_input_pdfs(folder_name):
    folder_path = 'input_pdfs/' + folder_name
    input_folder_path = 'inputs/' + folder_name

    # skip if the txts already exists
    if os.path.exists(input_folder_path):
        files_in_folder1 = {os.path.basename(f).replace('.pdf', '.txt') for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))}
        files_in_folder2 = {os.path.basename(f) for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))}
        if files_in_folder1 == files_in_folder2:
            print(f"All PDFs in {folder_path} have alredy been processed. No new files found.")
            return

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            # Open the PDF file
            with open(os.path.join(folder_path, file_name), 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                # Preprocess text
                common_index, common_lengths = find_common_index(reader)
                text_list = extract_text_list(reader, common_index, common_lengths)
                text = preprocess_text(text_list)
                ascii_text = unidecode.unidecode(text)
                # Save extracted text to .txt file
                text_dir = 'inputs/' + folder_name + '/'
                save_text_to_file(ascii_text, text_dir, file_name)
