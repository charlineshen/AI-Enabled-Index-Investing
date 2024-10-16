import chromadb

CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000


# Initialize the Chroma client and get the collection
client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
collection_name = 'semantic-split-collection'

collection = client.get_collection(name=collection_name)

# Use the peek function to view the top 5 records
top_5_records = collection.get()

print(type(top_5_records))
# dict_keys(['ids', 'embeddings', 'metadatas', 'documents', 'data', 'uris', 'included'])
print(top_5_records.keys())
print(top_5_records)

for key, values in top_5_records.items():
    if key in ['ids', 'metadatas', 'included']:
        print(key)
        print(f"One entry for {key}: {values[0]}")
