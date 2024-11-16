import os
import pandas as pd
import glob
import hashlib
import chromadb
from openai import OpenAI
from semantic_splitter import SemanticChunker
from tqdm import tqdm

# OpenAI API setup 
# TODO: REPLACE with Gemini/Copilot API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)
GENERATIVE_MODEL = "gpt-4o"

# TODO: REPLACE with Elasticsearch set up
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000

# Intermediate input/output folders
INPUT_FOLDER = "inputs" # save pdf-converted text files
OUTPUT_FOLDER = "outputs" # save chunked and embeded json files
QUESTION_FOLDER = 'question_templates' # folder containing question templates

# LLM configuration for content generation
generation_config = {
    "max_output_tokens": 4096,  # Maximum number of tokens for output
    "temperature": 0.1,  # Control randomness in output
    "top_p": 0.3,  # Percent of words to consider during nucleus sampling
}

# The system prompt to the LLM
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in ETF knowledge. You will be asked an ETF-related question and be given some reference text chunks. When answering, make sure your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in the provided chunks.

Remember:
- Your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside of the given text chunks.
- Be concise in your response while ensuring you cover all relevant information from the chunks.
"""

# The prompt appended to each question to the LLM
UNIFORM_PROMPT = """
You will be given a query and some contexts below. When answering a query, make sure you:
- Carefully read all the numbered context chunks provided.
- Identify the most relevant information from these numbered chunks to address the query's question.
- Formulate your response using only the information found in the given chunks.
- If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
- If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.
- Keep track of the numbers of all the chunks you used to formulate your answer.
- It is very important that you format your response using this pattern: "Answer: <X>.\nSource Chunks: <Y>.", where <Y> contains the numbers of chunks separated by commas. If no chunk provides useful information, <Y> should be None.
"""

####################
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased", cache_dir="model")
model = AutoModel.from_pretrained("google/mobilebert-uncased", cache_dir="model")

# Ensure the model is in evaluation mode
model.eval()

def generate_one_embedding(text):
	inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
	with torch.no_grad():
		output = model(**inputs, output_hidden_states=True)
		cls_embedding = output.last_hidden_state[:, 0, :]
		cls_embedding = torch.nn.functional.normalize(cls_embedding).cpu().squeeze().numpy()
	return cls_embedding
####################

# TODO: REPLACE with other desired embeddin
# Generate the embedding for a single piece of text using OpenAI's embedding model
# def generate_one_embedding(text, model="text-embedding-3-small"):
# 	return client.embeddings.create(input = [text], model=model).data[0].embedding

# Generate the embeddings for many chunks using OpenAI's embedding model
def generate_all_embeddings(chunks):
	embeddings = []
	for text in tqdm(chunks, "Embedding Chunks"):
		embeddings.append(generate_one_embedding(text))
	return embeddings

# Insert text embeddings into chromadb collection
def load_text_embeddings(df, collection, batch_size=32):
	# Generate ids based on document titles and dataframe index
	df["id"] = df.index.astype(str)
	hashed_titles = df["title"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
	df["id"] = hashed_titles + "-" + df["id"]

	# Preprocess metadata
	metadata = {
		"title": df["title"].tolist()[0],
		"zip_name": df["zip_name"].tolist()[0]
	}
	
	# Process data in batches
	total_inserted = 0
	for i in range(0, df.shape[0], batch_size):
		# Create a copy of the batch and reset the index
		batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)
		# Save dataframe content into collection
		collection.add(
			ids=batch["id"].tolist(),
			documents=batch["chunk"].tolist() ,
			metadatas=[metadata for _ in batch["title"].tolist()],
			embeddings=batch["embedding"].tolist()
		)
		total_inserted += len(batch)

	print(f"Finished inserting {total_inserted} items into collection '{collection.name}'")


def chunk(input_folder_name):
	# Make dataset folders
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)

	# Get the list of text file
	text_files = glob.glob(os.path.join(INPUT_FOLDER, input_folder_name, "*.txt"))

	# Process
	for text_file in text_files:
		print("Processing file:", text_file)
		filename = os.path.basename(text_file)
		title_name = filename.split(".")[0]
		if check_duplicates(title_name):
			print(f"File: {title_name} already exists in the database. Skipping...")
			continue

		zip_folder_name = os.path.basename(os.path.dirname(text_file))

		with open(text_file) as f:
			input_text = f.read()

		text_chunks = None

		# Initialize the splitter
		text_splitter = SemanticChunker(
			########
			embedding_function=generate_all_embeddings,
			########
			breakpoint_threshold_type="percentile",
			breakpoint_threshold_amount=85
		)

		# Perform splitting
		text_chunks = text_splitter.create_documents([input_text])
		text_chunks = [doc.page_content for doc in text_chunks if doc.page_content.strip()]
		print("Number of chunks:", len(text_chunks))

		if text_chunks is not None:
			# Save the chunks
			data_df = pd.DataFrame(text_chunks, columns=["chunk"])
			data_df["title"] = title_name
			data_df["zip_name"] = zip_folder_name

			jsonl_filename = os.path.join(OUTPUT_FOLDER, f"chunks-semantic-split-{title_name}.jsonl")
			with open(jsonl_filename, "w") as json_file:
				json_file.write(data_df.to_json(orient='records', lines=True))


def embed():
	# Get the list of chunk files
	jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"chunks-semantic-split-*.jsonl"))
	print("Number of files to process:", len(jsonl_files))

	# Generate embeddings for the chunks
	for jsonl_file in jsonl_files:
		print("\nProcessing file:", jsonl_file)

		data_df = pd.read_json(jsonl_file, lines=True)
		chunks = data_df["chunk"].values
		embeddings = generate_all_embeddings(chunks)
		data_df["embedding"] = embeddings

		# Save the embeddings
		jsonl_filename = jsonl_file.replace("chunks-","embeddings-")
		with open(jsonl_filename, "w") as json_file:
			json_file.write(data_df.to_json(orient='records', lines=True))


# TODO: REPLACE with Elasticsearch
def load():
	# Connect to chroma DB
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"semantic-split-collection"

	try:
		# Try to retrieve the existing collection by name
		collection = client.get_collection(name=collection_name)
		print(f"Retrieved existing collection '{collection_name}'")
	except Exception:
		# If the collection doesn't exist, create a new one
		collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
		print(f"Collection '{collection_name}' does not exist. Created new empty collection '{collection_name}'")

	# Get the list of embedding files
	jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"embeddings-semantic-split-*.jsonl"))
	print("Number of files to process:", len(jsonl_files))

	# Process
	for jsonl_file in jsonl_files:
		print("Processing file:", jsonl_file)
		data_df = pd.read_json(jsonl_file, lines=True)

		# Load data
		load_text_embeddings(data_df, collection)

# TODO: REPLACE with Elasticsearch
def query(title, question):
	# Connect to chroma DB
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"semantic-split-collection"

	# Generate the embedding of the question
	query_embedding = generate_one_embedding(question)

	# Get the collection
	collection = client.get_collection(name=collection_name)

	# retrieve chunks that come from the corresponding PDF
	collection_filtered = collection.get(where={"title": title})
	n_chunks = len(collection_filtered["ids"])

	try:
		results = collection.query(
			query_embeddings=[query_embedding],
			where={"title": title},
			n_results= min(int(n_chunks**0.4*2), 20) # Extract at most 20 chunks to limit context length
		)
	except Exception as e:
		print("error occurred:", str(e))

	return results

# TODO: REPLACE with Gemini/Copilot API
# Generate a GPT response using OpenAI's API based on the context chunks provided.
def generate_gpt_response(query, context_chunks):
	prompt = f"""
	{UNIFORM_PROMPT}

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

	# Open and read the Excel file
	data = pd.read_excel(QUESTION_FOLDER + '/' + input_file)

	# Extract questions from the second column
	questions = data.iloc[:, 1].tolist()  # Assuming the question is in the second column

    # Print the number of questions and answers to verify they match
	print(f"Number of questions: {len(questions)}")
	return questions

def chat(input_folder_name, input_question_file_name, output_file):
	questions = read_q(input_question_file_name)
	
	# Save results to an Excel file
	df = pd.DataFrame()
	for title in os.listdir(f"{INPUT_FOLDER}/{input_folder_name}"):
		if not title.endswith('.txt'):
			continue
		print("Processing document:", title)
		title = title.split(".")[0]
		singledoc_results = []
		for i in tqdm(range(len(questions))):
			query = questions[i]
		
			# Get the query, chunk, and actual response from the chat agent
			query, response, source_chunks = chat_agent(title, query)
			# Store results
			singledoc_results.append({
				f'{title}-response': response,
				f'{title}-source_chunks': source_chunks
			})

		df_singledoc = pd.DataFrame(singledoc_results)
		df = pd.concat([df, df_singledoc], axis=1)
		
	df.index = pd.Index(questions)
	df.to_excel(output_file, index=True)
	print(f"Results saved to '{output_file}'.")


def chat_agent(title, question):
	# Query relevant chunks
	results = query(title, question)
	numbered_results = [f"{i + 1}.\n{doc}" for i, doc in enumerate(results['documents'][0])]
	context_chunks = "\n--------------------------------------------------------------------------------------\n".join(numbered_results)

	# Generate a response using OpenAI GPT
	response_text = generate_gpt_response(question, context_chunks)

	# Get source chunk's number from the response text, if available, add to the response
	try:
		source_chunks_nums = response_text.split("Source Chunks: ")[1].split(".")[0].split(",")
		source_chunks_nums = [int(num) for num in source_chunks_nums]
		source_chunks = "\n--------------------------------------------------------------------------------------\n".join([results["documents"][0][i - 1] for i in source_chunks_nums])
	except:
		source_chunks = "Source Chunks: None"
		pass

	response_text = response_text.split("Source Chunks: ")[0]
	return question, response_text, source_chunks

# TODO: REPLACE with Elasticsearch
def check_duplicates(title):
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
	collection_name = f"semantic-split-collection"
	# Get the collection
	try:
		collection = client.get_collection(name=collection_name)
	except:
		return False

	# retrieve chunks that come from the corresponding PDF
	collection_filtered = collection.get(where={"title": title})
	n_chunks = len(collection_filtered["ids"])

	if n_chunks == 0:
		return False
	return True
