# rag.py

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


VECTOR_DIR = "vectorstore"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    collection_name="support_knowledge",
    embedding_function=embeddings,
    persist_directory=VECTOR_DIR
)


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)


llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)


prompt = ChatPromptTemplate.from_template("""
You are a customer support assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I don't have enough information in the knowledge base to answer that."

Do not invent information.

Always mention the source documents used.

Context:
{context}

Question:
{question}

""")


def rag_chatbot(question: str):

    documents = retriever.invoke(question)

    if not documents:
        return {
            "answer": "I don't have enough information in the knowledge base.",
            "sources": []
        }

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in documents
    })

    return {
        "answer": response.content,
        "sources": sources
    }


if __name__ == "__main__":

    result = rag_chatbot(
        "How can I reset my password?"
    )

    print(result["answer"])
    print("\nSources:")

    for source in result["sources"]:
        print("-", source)