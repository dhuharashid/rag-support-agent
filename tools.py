# tools.py

from langchain.tools import tool

from rag import rag_chatbot


@tool
def knowledge_base_search(question: str) -> str:
    """
    Search the company's internal knowledge base and answer
    questions using trusted support documentation.
    """

    result = rag_chatbot(question)

    sources = ", ".join(result["sources"])

    return (
        f"Answer:\n{result['answer']}\n\n"
        f"Sources: {sources}"
    )