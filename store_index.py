import hashlib
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

PROJECT_DIR = Path(__file__).resolve().parent
INDEX_NAME = "medai"
EMBEDDING_DIMENSION = 384


def _stable_chunk_ids(documents):
    page_chunk_counts = Counter()
    ids = []
    for document in documents:
        source = str(document.metadata.get("source", "")).replace("\\", "/")
        page = str(document.metadata.get("page", ""))
        chunk_number = page_chunk_counts[(source, page)]
        page_chunk_counts[(source, page)] += 1
        identity = f"{source}\0{page}\0{chunk_number}\0{document.page_content}"
        ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    return ids


def main():
    load_dotenv(PROJECT_DIR / ".env")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    data_dir = PROJECT_DIR / "data"
    extracted_docs = load_pdf_files(str(data_dir))
    if not extracted_docs:
        raise RuntimeError(f"No PDF pages were loaded from {data_dir}")

    contentful_docs = [
        document for document in extracted_docs
        if document.page_content and document.page_content.strip()
    ]
    if not contentful_docs:
        raise RuntimeError("No text could be extracted from the PDFs in data/")

    pages_by_source = Counter(
        str(document.metadata.get("source", "unknown"))
        for document in contentful_docs
    )
    for source in sorted({str(doc.metadata.get("source", "unknown")) for doc in extracted_docs}):
        if source not in pages_by_source:
            print(f"Skipping PDF with no extractable text: {source}")

    text_chunks = text_split(filter_to_minimal_docs(contentful_docs))
    if not text_chunks:
        raise RuntimeError("Text was extracted, but no chunks were created")
    chunk_ids = _stable_chunk_ids(text_chunks)

    print(f"Extracted text from {len(contentful_docs)} of {len(extracted_docs)} pages.")
    print(f"Created {len(text_chunks)} chunks.")
    for source in sorted(pages_by_source):
        print(f"  {source}: {pages_by_source[source]} pages")

    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)
    count_before = index.describe_index_stats().total_vector_count
    print(f"Pinecone records before upsert: {count_before}")

    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        ids=chunk_ids,
    )

    target_count = max(count_before, len(chunk_ids))
    deadline = time.monotonic() + 60
    count_after = index.describe_index_stats().total_vector_count
    while count_after < target_count and time.monotonic() < deadline:
        time.sleep(2)
        count_after = index.describe_index_stats().total_vector_count

    print(f"Pinecone records after upsert: {count_after}")
    if count_after < target_count:
        raise RuntimeError(
            f"Pinecone record count did not reach {target_count} within 60 seconds"
        )


if __name__ == "__main__":
    main()
