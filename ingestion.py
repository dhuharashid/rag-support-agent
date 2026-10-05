# ingestion.py

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_DIR = "data"
VECTOR_DIR = "vectorstore"


def build_vector_database():

    documents = []

    for pdf_file in Path(DATA_DIR).glob("*.pdf"):

        loader = PyPDFLoader(str(pdf_file))

        docs = loader.load()
        for doc in docs:
            if not doc.page_content.strip():
                print(f"Warning: Empty page in {pdf_file.name}")
            else:
                print(
                f"Extracted text from {pdf_file.name}: "
                f"{len(doc.page_content)} characters"
                )

        for doc in docs:
            doc.metadata["source"] = pdf_file.name

        documents.extend(docs)

    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,
        collection_name="support_knowledge"
    )

    print("Vector database created.")

    return vectorstore


if __name__ == "__main__":
    build_vector_database()