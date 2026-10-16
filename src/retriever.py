
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

INDEX_DIR = Path("chroma_index")
COLLECTION_NAME = "movie_research"


def search(query, k=3):
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
    )

    for index, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][index]

        print(f"\nResult {index + 1}")
        print(f"Movie: {metadata['title']}")
        print(f"Source: {metadata['source']}")
        print(f"Passage: {text[:500]}...")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])

    if not query:
        query = "Which movie explores artificial intelligence and consciousness?"

    search(query)
