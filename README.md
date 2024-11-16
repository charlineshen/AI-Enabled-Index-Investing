# AI-Enabled Index Investing

![Pipeline Overview](demo.png)

Run the following commands to pull the code locally:
```bash
git clone https://github.com/charlineshen/AI-Enabled-Index-Investing.git
cd AI-Enabled-Index-Investing
```

### Prepare the Following before Using the Tool:
1. A folder containing the index documents pdfs. Put this folder under the `input_pdfs` directory.
2. An Excel file containing the question template, with the questions located in the **second** column. Put this excel file under the `question_templates` folder.
3. A secret folder containing the   

1. Git can be installed [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
2. Docker Desktop can be installed [here](https://www.docker.com/products/docker-desktop/).
3. For Windows users, commands need to be run in a Linux subsystem like [Ubuntu WSL](https://ubuntu.com/desktop/wsl). For Mac users, commands can be run directly in the system terminal.

### Instructions to Use the Tool:
1. Launch Docker desktop.
2. Run the following command to start the Docker container.
    ```bash
    sh docker-shell.sh
    ```
3. Run the following command to process the documents and questions:
    ```bash
    python controller.py <index_folder_name> <question_template_excel_name>
    ```
    Specifically, the following steps will be performed:
    * process the index PDFs in a given folder, preprocess them and save them as txt files, and save the processed txt files under `inputs/` folder.
    * chunk text files into smaller pieces using semantic spilt algorithm.
    * generate embeddings for each chunk and save them in a local ChromaDB instance.
    * generate a comparison table, where the rows will be questions, and the columns will be answers and citations corresponding to each index document.
    * An example command is `python controller.py test test.xlsx`.

### Notes:
1. To avoid cost and duplicated effort, we will NOT process files with exact same name twice. If the content of files changed, please rename it.
2. The llm-rag-chromadb container is designed to be a persistent and ongoing service to host the Chroma database, so it does not stop upon exit. If necessary, shut it down manually, and the saved chunks and embeddings will be cleared.
3. If you are having trouble with ChromaDB HTTP connections, switch from `chromadb = "0.5.18"` to `chromadb = "0.5.11"` in `Pipfile`.
