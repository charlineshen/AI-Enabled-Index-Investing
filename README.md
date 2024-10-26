# AI-enabled-Index-Investing
* To process a zip file containing PDF documents, run `python data_process.py`. This will unzip the file, convert the PDF documents to txt with some data preprocessing, and save the processed txt files under `inputs/`. To specify the zip file to process, change the `zip_path` variable under the main function in `data_process.py`.
* To chunk files under `inputs/` and generate embeddings, run `python cli.py --chunk --embed --load`. By default it uses semantic splitting.
* To generate an answer table, run `python cli.py --chat`. To specify the zip file to ask questions to, change the `zipname` variable under the `chat` function in `cli.py`.
* To test against "MSCI Select ESG Screened Indexes Methodology" and generate some example answers, run `python auto_evaluator.py`.

*Note on Docker container: The llm-rag-chromadb container is designed to be a persistent and ongoing service, so it does not stop upon exit. It should be manually shut down if necessary.*