# AI-Enabled Index Investing

![Pipeline Overview](demo.png)

First, run the following commands to pull the code locally:
```bash
git clone https://github.com/charlineshen/AI-Enabled-Index-Investing.git
cd AI-Enabled-Index-Investing
```

### Prepare the Following before Using the Tool:
1. A folder containing the index documents pdfs. Put this folder under the `input_pdfs` directory.
2. An Excel file containing the question template, with the questions located in the **second** column. Put this excel file under the `question_templates` folder.

### Instructions to Use the Tool:
1. Launch Docker desktop.
2. Run the following commands to  start the Docker container.
    ```bash
    sh docker-shell.sh
    ```
3. Run the following command to process the documents and questions:
    ```bash
    python controller.py <index_folder_name> <question_template_excel_name>
    ```
    Specifically, the following steps will be performed:
    * process the index PDFs in a given folder, preprocess them and save them as txt files, and save the processed txt files under `inputs/` folder.
    * chunk text files into smaller pieces using semantic spilt algorithm
    * generate embeddings foor each chunk and save them in a local ChromaDB instance
    * generate a comparison table, where the rows will be questions, and the columns will be answers and citations corresponding to each index document
    * An example command is `python controller.py test test.xlsx`.

### Notes:
1. Docker Desktop can be installed [here](https://www.docker.com/products/docker-desktop/).
2. For Windows users, commands need to be run in a Linux subsystem like [Ubuntu WSL](https://ubuntu.com/desktop/wsl). For Mac users, commands can be run directly in the system terminal.
3. To avoid cost and duplicated effort, we will NOT process files with exact same name twice. If the content of files changed, please rename it.
4. The llm-rag-chromadb container is designed to be a persistent and ongoing service to host the Chroma database, so it does not stop upon exit. If necessary, shut it down manually, and the saved chunks and embeddings will be cleaned.
