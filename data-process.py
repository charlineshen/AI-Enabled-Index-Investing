import PyPDF2
import os
import re
import unidecode

# Find the line index of header and footer
def find_common_index(pdf_path):
    page_line_lengths = []
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Only check the first 5 pages to find common lines
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

# Return a list of strings; each string is a page's content without header and footer
def extract_text_list(pdf_path, common_index, common_lengths):
    full_text = []
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        for page_num in range(len(reader.pages)):
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

# Returns a mega string of the entire PDF without cover, table of contents, contact us, and disclaimer
def preprocess_text(text_list):
    # Search for the first occurrence of the regex pattern      
    toc_pattern = re.compile(r'\d+[^\d]+(\d+)')
    toc_match = toc_pattern.search(text_list[1])
    if toc_match:
        # Extract the second numeric group as the start page
        start_page = int(toc_match.group(1)) - 1
    else:
        print("No matching pattern found. The Contents page of the PDF may have an uncommon format.")
    
    # Search for and remove the contacts page and the disclaimer page
    contacts_pattern = re.compile(r'\+ 1 \d{3} \d{3} \d{4}') # US phone number
    disclaimer_pattern = re.compile(r'This document and all of the information contained in it') # disclaimer-like content
    contacts, disclaimer = None, None
    for page_text in text_list: # search for the last occurence
        if contacts_pattern.search(page_text):
            contacts = page_text # save content instead of index to avoid indexing errors when removing two items
        if disclaimer_pattern.search(page_text):
            disclaimer = page_text
    if contacts:
        text_list.remove(contacts)
    if disclaimer:
        text_list.remove(disclaimer)

    # Merge list of text into long string
    text = "\n".join(text_list[start_page:])
    return text

def save_text_to_file(text, pdf_path):
    # Extract the filename without extension
    text_dir = os.path.dirname(pdf_path).replace('index-doc', 'input-datasets/books')
    base_name = os.path.basename(pdf_path).replace('.pdf', '.txt')
    # Construct the new path for the .txt file
    txt_path = os.path.join(text_dir, base_name)
    print('txt_path: ' + txt_path)

    
    # Write the extracted text to the file
    with open(txt_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)
    
    return txt_path

if __name__ == '__main__':
    pdf_path = "C:\Study@Harvard\Capstone\AI-enabled-Index-Investing\index-doc\MSCI_Select_ESG_Screened_Indexes_Methodology_20230519.pdf"
    common_index, common_lengths = find_common_index(pdf_path)
    text_list = extract_text_list(pdf_path, common_index, common_lengths)
    text = preprocess_text(text_list)
    ascii_text = unidecode.unidecode(text)
    saved_file_path = save_text_to_file(ascii_text, pdf_path)
    print(f"Text saved to: {saved_file_path}")