import PyPDF2
import os
import unidecode

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
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
    pdf_path = "/Users/jingwen/Desktop/2024 Fall/AC 297/AI-enabled-Index-Investing/index-doc/MSCI_Select_ESG_Screened_Indexes_Methodology_20230519.pdf"
    text = extract_text_from_pdf(pdf_path)
    ascii_text = unidecode.unidecode(text)
    saved_file_path = save_text_to_file(ascii_text, pdf_path)
    print(f"Text saved to: {saved_file_path}")