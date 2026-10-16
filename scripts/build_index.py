
import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCUMENT_DIR = Path("data/documents")
INDEX_DIR = Path("chroma_index")
COLLECTION_NAME = "movie_research"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents():
    documents = []

    for path in sorted(DOCUMENT_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not data.get("text", "").strip():
            continue

        documents.append(data)

    return documents


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    documents = load_documents()

    if not documents:
        raise RuntimeError("No documents found")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunk_ids = []
    chunk_texts = []
    chunk_metadata = []

    for document in documents:
        chunks = splitter.split_text(document["text"])

        for index, chunk in enumerate(chunks):
            chunk_ids.append(
                f"{document['title']}:{index}"
            )
            chunk_texts.append(chunk)
            chunk_metadata.append({
                "title": document["title"],
                "source": document["url"],
                "chunk_index": index,
            })

        print(f"{document['title']}: {len(chunks)} chunks")

    print(f"Total chunks: {len(chunk_texts)}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectors = embeddings.embed_documents(chunk_texts)

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    collection.upsert(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=vectors,
        metadatas=chunk_metadata,
    )

    print(f"Stored {collection.count()} chunks in Chroma")


if __name__ == "__main__":
    main()
