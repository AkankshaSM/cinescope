
import gc
import json
import os
import shutil
import uuid
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

DOCUMENT_DIR = ROOT_DIR / "data/documents"
MOVIES_PATH = ROOT_DIR / "data/raw/movies.json"
INDEX_DIR = ROOT_DIR / "chroma_index"

COLLECTION_NAME = "movie_research"
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 50


def load_documents():
    movies = json.loads(
        MOVIES_PATH.read_text(encoding="utf-8")
    )

    valid_ids = {movie["id"] for movie in movies}
    documents = {}

    for path in sorted(DOCUMENT_DIR.glob("*.json")):
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        movie_id = data.get("tmdb_id")

        if movie_id is None:
            print(f"Skipping legacy document: {path.name}")
            continue

        if movie_id not in valid_ids:
            print(f"Skipping unknown movie: {path.name}")
            continue

        if not data.get("text", "").strip():
            print(f"Skipping empty document: {path.name}")
            continue

        if movie_id in documents:
            raise ValueError(
                f"Duplicate document for TMDB ID {movie_id}"
            )

        documents[movie_id] = data

    return list(documents.values())


def prepare_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for document in documents:
        movie_id = document["tmdb_id"]

        text_chunks = splitter.split_text(
            document["text"]
        )

        for index, text in enumerate(text_chunks):
            chunks.append({
                "id": f"{movie_id}:{index}",
                "text": text,
                "metadata": {
                    "tmdb_id": movie_id,
                    "title": document["title"],
                    "source": document["url"],
                    "wikidata_id": document.get(
                        "wikidata_id"
                    ) or "",
                    "chunk_index": index,
                },
            })

        print(
            f"{document['title']}: "
            f"{len(text_chunks)} chunks"
        )

    return chunks


def build_staging_index(chunks, staging_dir):
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(
        path=str(staging_dir)
    )

    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
        )

        total = len(chunks)

        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]

            texts = [item["text"] for item in batch]

            vectors = embeddings.embed_documents(texts)

            collection.upsert(
                ids=[item["id"] for item in batch],
                documents=texts,
                embeddings=vectors,
                metadatas=[
                    item["metadata"] for item in batch
                ],
            )

            print(
                f"Indexed {min(start + BATCH_SIZE, total)}"
                f"/{total} chunks"
            )

        count = collection.count()

        if count != total:
            raise RuntimeError(
                f"Index incomplete: {count}/{total}"
            )

        print(f"Staging index verified: {count} chunks")

    finally:
        del collection
        del client
        gc.collect()


def activate_index(staging_dir):
    backup_dir = INDEX_DIR.with_name(
        INDEX_DIR.name + ".backup"
    )

    if backup_dir.exists():
        raise RuntimeError(
            f"Backup directory already exists: {backup_dir}"
        )

    had_existing_index = INDEX_DIR.exists()

    if had_existing_index:
        INDEX_DIR.rename(backup_dir)

    try:
        staging_dir.rename(INDEX_DIR)

    except Exception:
        if had_existing_index and backup_dir.exists():
            backup_dir.rename(INDEX_DIR)
        raise

    print(f"Active index: {INDEX_DIR}")

    if had_existing_index:
        print(
            f"Previous index preserved at: {backup_dir}"
        )


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    documents = load_documents()

    if not documents:
        raise RuntimeError("No valid documents found")

    print(f"\nDocuments: {len(documents)}")

    chunks = prepare_chunks(documents)

    print(f"\nTotal chunks: {len(chunks)}")

    staging_dir = INDEX_DIR.with_name(
        INDEX_DIR.name + ".staging-" + uuid.uuid4().hex[:8]
    )

    try:
        build_staging_index(chunks, staging_dir)
        activate_index(staging_dir)

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("\nChroma index build complete")


if __name__ == "__main__":
    main()
