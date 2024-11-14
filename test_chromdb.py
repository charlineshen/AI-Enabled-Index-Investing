import chromadb
from openai import OpenAI
import os
from transformers import AutoTokenizer, AutoModel
import json

CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000

# OpenAI API setup
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

### Embedding ###
# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", cache_dir="model")
model = AutoModel.from_pretrained("albert-base-v2", cache_dir="model")
# tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir="model")
# model = LongformerModel.from_pretrained("allenai/longformer-base-4096", cache_dir="model")

# Ensure the model is in evaluation mode
model.eval()




def generate_query_embedding(text, model="text-embedding-3-small"):
	return client.embeddings.create(input = [text], model=model).data[0].embedding


def query(zip_name, title, question, method="semantic-split"):
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
	n_chunks = len(collection_filtered["ids"])
	unique_ids = len(set(collection_filtered["ids"]))
	print(f'title: {title}, n_chunks: {n_chunks}, unique_ids: {unique_ids}')

	# 2: Query based on embedding value + metadata filter
	# retrieve chunks based on embedding similarity compared to query
	results = collection.query(
		where={"title": title},
		query_embeddings=[query_embedding],
		n_results= n_chunks #min(int(n_chunks**0.4*2), 40) # about 10 out of 50 chunks, 24 out of 200 chunks
	)

	return results

def check_duplicates(title, method="semantic-split"):
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
	collection_name = f"{method}-collection"
	# Get the collection
	collection = client.get_collection(name=collection_name)

	# retrieve chunks that come from the corresponding PDF
	collection_filtered = collection.get(where={"title": title})
	n_chunks = len(collection_filtered["ids"])

	if n_chunks == 0:
		return False
	return True


if __name__ == "__main__": 
    
	# client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
	# collection_name = 'semantic-split-collection'
	# collection = client.get_collection(name=collection_name)
	# # Use the peek function to view the top 5 records
	# top_5_records = collection.get()
	
	# print(type(top_5_records))
	# dict_keys(['ids', 'embeddings', 'metadatas', 'documents', 'data', 'uris', 'included'])
	# print(top_5_records.keys())
	# print(top_5_records)

	
	# for key, values in top_5_records.items():
	# 	if key in ['metadatas']:
	# 		print(f"One entry for {key}: {value['title'] for value in values}")
	# 		print(len(values))
	


	############################################################### 
	# general4: Is there a free-float market capitalization adjustment?
	# general1: What are the criteria for a stock to be eligible for inclusion in the index?
	# general8: What is the Weighting Methodology?
	'''
	Check retrieval performance
	'''
	title = '1_MSCI_Global_Investable_Market_Indexes_Methodology_20240812'
	question = "What is the Weighting Methodology?"
	result = query('', title, question, method="semantic-split")
	print(len(result))
	
	json_path = "general4.json"

    # Save the dictionary as a JSON file
	with open(json_path, 'w', encoding='utf-8') as json_file:
		json.dump(result, json_file, indent=4)  # indent=4 for pretty formatting
	

	################################################################
	# '''
	# Test duplicates
	# '''
	# title = '3_MSCI_Select_ESG_Screened_Indexes_Methodology_20240209'
	# print(title, check_duplicates(title, method="semantic-split"))


	# title = '3_MSCI_Low_Carbon_SRI_Leaders_Indexes_Methodology_20240219'
	# print(title, check_duplicates(title, method="semantic-split"))
