import os
import argparse
import pandas as pd
import json
import time
import glob
import hashlib
import chromadb
import csv
from openai import OpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
from semantic_splitter import SemanticChunker
# import agent_tools

import torch
from transformers import BertTokenizer, BertModel

# Setup TODO
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gpt-4o"
INPUT_FOLDER = "inputs"
OUTPUT_FOLDER = "outputs"
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000

# model.cuda()  # uncomment it if you have a GPU

# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 4096,  # Maximum number of tokens for output
    "temperature": 0.25,  # Control randomne ss in output
    "top_p": 0.95,  # Use nucleus sampling
}
# Initialize the GenerativeModel with specific system instructions
# In your response, specify the source of the information you used to answer the query (e.g., the section of the text and the original text).
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in ETF knowledge. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.

When answering a query:
- Carefully read all the text chunks provided.
- Identify the most relevant information from these chunks to address the user's question.
- Formulate your response using only the information found in the given chunks.
- If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
- If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.
- In your response, follow the format of "Answer: xxx\n Source Text: xxx\n Section of Source Text: xxx"

Remember:
- Your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside of the given text chunks.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.
"""

### embedding
# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="model")
model = BertModel.from_pretrained('bert-base-uncased', cache_dir="model")

# Ensure the model is in evaluation mode
model.eval()

def embed_bert_cls(text):
    # Tokenize the input text and get the input IDs
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model to get outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings for the CLS token
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

def generate_query_embedding(query):
	return embed_bert_cls(query)

def generate_text_embeddings(chunks, batch_size=128):
	all_embeddings_new = []
	for i in range(0, len(chunks), batch_size):
		batch = chunks[i:i+batch_size]
		embeddings = [embed_bert_cls(text) for text in batch] 
		all_embeddings_new.extend([embedding for embedding in embeddings])
	print("=============================== new embeddings")
	print(len(all_embeddings_new), len(all_embeddings_new[0]))
	print("===============================")
	return all_embeddings_new
### end of embedding

def load_text_embeddings(df, collection, batch_size=128):
	# Generate ids
	df["id"] = df.index.astype(str)
	hashed_titles = df["title"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
	# hashed_zips = df["zip_name"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
	df["id"] = hashed_titles + "-" + df["id"]
	# df["id"] = hashed_titles+ hashed_zips  + "-" + df["id"]
	
	metadata = {
		"title": df["title"].tolist()[0],
		"zip_name": df["zip_name"].tolist()[0]
	}
	
	# Process data in batches
	total_inserted = 0
	for i in range(0, df.shape[0], batch_size):
		# Create a copy of the batch and reset the index
		batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

		ids = batch["id"].tolist()
		documents = batch["chunk"].tolist() 
		metadatas = [metadata for item in batch["title"].tolist()]
		embeddings = batch["embedding"].tolist()

		collection.add(
			ids=ids,
			documents=documents,
			metadatas=metadatas,
			embeddings=embeddings
		)
		total_inserted += len(batch)
		print(f"Inserted {total_inserted} items...")

	print(f"Finished inserting {total_inserted} items into collection '{collection.name}'")


def chunk(method="semantic-split"):
	print("chunk()")

	# Make dataset folders
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)

	# Get the list of text file
	text_files = glob.glob(os.path.join(INPUT_FOLDER, "**", "*.txt"), recursive=True)
	print("Number of files to process:", len(text_files))

	# Process
	for text_file in text_files:
		print("Processing file:", text_file)
		filename = os.path.basename(text_file)
		title_name = filename.split(".")[0]
		zip_folder_name = os.path.basename(os.path.dirname(text_file))

		with open(text_file) as f:
			input_text = f.read()

		text_chunks = None
		if method == "char-split":
			chunk_size = 350
			chunk_overlap = 20
			# Init the splitter
			text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap, separator='', strip_whitespace=False)

			# Perform the splitting
			text_chunks = text_splitter.create_documents([input_text])
			text_chunks = [doc.page_content for doc in text_chunks]
			print("Number of chunks:", len(text_chunks))

		elif method == "recursive-split":
			chunk_size = 350
			# Init the splitter
			text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size)

			# Perform the splitting
			text_chunks = text_splitter.create_documents([input_text])
			text_chunks = [doc.page_content for doc in text_chunks]
			print("Number of chunks:", len(text_chunks))

		elif method == "semantic-split":
			# Init the splitter
			text_splitter = SemanticChunker(
				embedding_function=generate_text_embeddings,
				breakpoint_threshold_type="percentile",
				breakpoint_threshold_amount=80
			)
			# Perform the splitting
			text_chunks = text_splitter.create_documents([input_text])

			text_chunks = [doc.page_content for doc in text_chunks if doc.page_content.strip()]
			print("Number of chunks:", len(text_chunks))

		if text_chunks is not None:
			# Save the chunks
			data_df = pd.DataFrame(text_chunks,columns=["chunk"])
			data_df["title"] = title_name
			data_df["zip_name"] = zip_folder_name
			print("Shape:", data_df.shape)
			print(data_df.head())

			jsonl_filename = os.path.join(OUTPUT_FOLDER, f"chunks-{method}-{title_name}.jsonl")
			with open(jsonl_filename, "w") as json_file:
				json_file.write(data_df.to_json(orient='records', lines=True))


def embed(method="semantic-split"):
	print("embed()")

	# Get the list of chunk files
	jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"))
	print("Number of files to process:", len(jsonl_files))

	# Process
	for jsonl_file in jsonl_files:
		print("Processing file:", jsonl_file)

		data_df = pd.read_json(jsonl_file, lines=True)
		print("Shape:", data_df.shape)
		print(data_df.head())

		chunks = data_df["chunk"].values
		if method == "semantic-split":
			embeddings = generate_text_embeddings(chunks, EMBEDDING_DIMENSION, batch_size=15)
		else:
			embeddings = generate_text_embeddings(chunks, EMBEDDING_DIMENSION, batch_size=100)
		data_df["embedding"] = embeddings

		# Save 
		print("Shape:", data_df.shape)
		print(data_df.head())

		jsonl_filename = jsonl_file.replace("chunks-","embeddings-")
		with open(jsonl_filename, "w") as json_file:
			json_file.write(data_df.to_json(orient='records', lines=True))


def load(method="semantic-split"):
	print("load()")

	# Connect to chroma DB
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"{method}-collection"
	print("Creating collection:", collection_name)

	try:
		# Clear out any existing items in the collection
		client.delete_collection(name=collection_name)
		print(f"Deleted existing collection '{collection_name}'")
	except Exception:
		print(f"Collection '{collection_name}' did not exist. Creating new.")

	collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
	print(f"Created new empty collection '{collection_name}'")
	print("Collection:", collection)

	# Get the list of embedding files
	jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"))
	print("Number of files to process:", len(jsonl_files))

	# Process
	for jsonl_file in jsonl_files:
		print("Processing file:", jsonl_file)

		data_df = pd.read_json(jsonl_file, lines=True)
		print("Shape:", data_df.shape)
		print(data_df.head())

		# Load data
		load_text_embeddings(data_df, collection)

def query(zip_name, title, question, method="semantic-split"):
	# print("query()")

	# Connect to chroma DB
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"{method}-collection"

	query_embedding = generate_query_embedding(question)
	# print("Embedding values:", query_embedding)

	# Get the collection
	collection = client.get_collection(name=collection_name)

	# retrieve chunks that come from the corresponding PDF
	collection_filtered = collection.get(where={"title": title})
	n_chunks = len(collection_filtered)

	# # 1: Query based on embedding value 
	# results = collection.query(
	# 	query_embeddings=[query_embedding],
	# 	n_results=10
	# )

	# 2: Query based on embedding value + metadata filter
	# retrieve chunks based on embedding similarity compared to query
	results = collection.query(
		where={"title": title},
		query_embeddings=[query_embedding],
		n_results=int(n_chunks**0.6) # about 10 out of 50 chunks, 24 out of 200 chunks
	)

	# 3: Query based on embedding value + lexical search filter
	# search_string = "Italian"
	# results = collection.query(
	# 	query_embeddings=[query_embedding],
	# 	n_results=10,
	# 	where_document={"$contains": search_string}
	# )
	# print("Query:", query)
	# print("\n\nResults:", results)

	return results


def generate_gpt_response(query, context_chunks):
	"""
	Generate a GPT response using OpenAI's API based on the context chunks provided.
	"""
	prompt = f"""
	{SYSTEM_INSTRUCTION}

	Query: {query}
	Context:
	{context_chunks}
	"""
	response = client.chat.completions.create(
			model=GENERATIVE_MODEL,
			messages=[
				{"role": "system", "content": SYSTEM_INSTRUCTION},
				{"role": "user", "content": prompt}
			],
			temperature=generation_config['temperature'],
			max_tokens=generation_config['max_output_tokens'],
			top_p=generation_config['top_p'])
	return response.choices[0].message.content

def read_q(input_file):
    questions = []
    # Open and read the CSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            questions.append(row[2])  # Question is in the second column

    # Print the number of questions and answers to verify they match
    print(f"\nNumber of questions: {len(questions)}")
    return questions

def chat():
	input_question_file = "questions.csv"
	output_file = "results.csv"
	zip_name = 'MSCI_indexes_small'
	questions = read_q(input_question_file)
	
	# Save results to an Excel file
	df = pd.DataFrame()
	for title in os.listdir(f"{INPUT_FOLDER}/{zip_name}"):
		print("Processing document:", title)
		title = title.split(".")[0]
		singledoc_results = []
		for i in range(len(questions)):
			query = questions[i]
			# print("================= query: ", query)
		
			# Get the query, chunk, and actual response from the chat agent
			query, chunk, response = chat_agent(zip_name, title, query)
			# Store results
			singledoc_results.append({
					f'{title}-response': response,
					f'{title}-chunks': chunk
				})

		df_singledoc = pd.DataFrame(singledoc_results)
		df = pd.concat([df, df_singledoc], axis=1)
		
	df.index = pd.Index(questions)
	df.to_csv(output_file, index=True)
	print(f"Results saved to '{output_file}'.")

def chat_agent(zip_name, title, question, method="semantic-split"):
	# print("chat()")
	# print("=====================", question, method)
	
	# Query relevant chunks
	results = query(zip_name, title, question, method)

	numbered_results = [f"{i + 1}.\n{doc}" for i, doc in enumerate(results['documents'][0])]
	formatted_results = "\n--------------------------------------------------------------------------------------\n".join(numbered_results)
	# print_output = f"{question}\n\n============================================RETRIEVED TEXT============================================n{formatted_results}"
	# print("============================================INPUT PROMPT============================================\n", print_output)

	# Prepare input prompt for OpenAI GPT model
	# context_chunks = "\n".join(results["documents"][0])
	# print(f"Context chunks: {context_chunks}")

	context_chunks = formatted_results

	# Generate a response using OpenAI GPT
	response_text = generate_gpt_response(question, context_chunks)

	# Print the GPT output
	# print(f"\n============================================GPT RESPONSE============================================\n{response_text}\n")
	return question, context_chunks, response_text


def main(args=None):
	# print("CLI Arguments:", args)

	if args.chunk:
		chunk(method=args.chunk_type)

	if args.embed:
		embed(method=args.chunk_type)

	if args.load:
		load(method=args.chunk_type)

	if args.query:
		query(method=args.chunk_type)

	if args.chat:
		chat(method=args.chunk_type)

	# if args.agent:
	# 	agent(method=args.chunk_type)


if __name__ == "__main__":
	# Generate the inputs arguments parser
	# if you type into the terminal '--help', it will provide the description
	parser = argparse.ArgumentParser(description="CLI")

	parser.add_argument(
		"--chunk",
		action="store_true",
		help="Chunk text",
	)
	parser.add_argument(
		"--embed",
		action="store_true",
		help="Generate embeddings",
	)
	parser.add_argument(
		"--load",
		action="store_true",
		help="Load embeddings to vector db",
	)
	parser.add_argument(
		"--query",
		action="store_true",
		help="Query vector db",
	)
	parser.add_argument(
		"--chat",
		action="store_true",
		help="Chat with LLM",
	)
	parser.add_argument(
		"--agent",
		action="store_true",
		help="Chat with LLM Agent",
	)
	parser.add_argument("--chunk_type", default="semantic-split", help="char-split | recursive-split | semantic-split")

	args = parser.parse_args()

	main(args)