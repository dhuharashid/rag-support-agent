# RAG-Based AI Support Agent

A Retrieval-Augmented Generation (RAG) support agent built with **LangGraph, LangChain, ChromaDB, Hugging Face embeddings, Ollama, and Streamlit**.

The application allows users to ask questions about a support knowledge base. Relevant information is retrieved from uploaded documents and provided to an LLM so that responses are grounded in the available knowledge rather than relying only on the model's general knowledge.

## 🚀 Project Overview

This project demonstrates how a traditional RAG chatbot can be extended into an **agentic workflow using LangGraph**.

The system:

* Ingests PDF-based support documentation
* Splits documents into searchable chunks
* Generates vector embeddings using Hugging Face
* Stores embeddings in ChromaDB
* Retrieves relevant documents for a user query
* Exposes the RAG pipeline as a tool
* Uses LangGraph to orchestrate the support agent
* Generates responses using a local Ollama LLM
* Provides a Streamlit interface for interaction

## 🏗️ Architecture

```text
                Support PDF
                    │
                    ▼
          Document Ingestion
                    │
                    ▼
           Text Chunking
                    │
                    ▼
       Hugging Face Embeddings
                    │
                    ▼
             ChromaDB
          Vector Database
                    │
                    ▼
             RAG Retriever
                    │
                    ▼
        Knowledge Base Tool
                    │
                    ▼
          LangGraph Agent
                    │
                    ▼
             Ollama LLM
                    │
                    ▼
            Support Response
                    │
                    ▼
            Streamlit UI
```

## 🧠 RAG Pipeline

The document processing pipeline is implemented in `ingestion.py`.

1. PDF documents are loaded using `PyPDFLoader`.
2. Documents are split using `RecursiveCharacterTextSplitter`.
3. Hugging Face embeddings are generated using `HuggingFaceEmbeddings`.
4. The embeddings are stored in ChromaDB.
5. User questions are converted into embeddings.
6. Relevant document chunks are retrieved from the vector database.
7. Retrieved context is passed to the LLM.
8. The model generates a response grounded in the retrieved knowledge.

## 🤖 Agentic Workflow

The RAG system is integrated into a LangGraph-based support agent.

```text
User Query
    │
    ▼
LangGraph Support Agent
    │
    ▼
Knowledge Base Search Tool
    │
    ▼
RAG Retrieval
    │
    ▼
Relevant Context
    │
    ▼
Ollama LLM
    │
    ▼
Final Answer
```

The knowledge base search functionality is exposed as a tool, allowing the agent to use the retrieval system as part of its reasoning workflow.

## 🛠️ Technologies Used

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Application development |
| LangGraph    | Agent orchestration     |
| LangChain    | LLM/RAG components      |
| ChromaDB     | Vector database         |
| Hugging Face | Text embeddings         |
| Ollama       | Local LLM inference     |
| Streamlit    | User interface          |
| PyPDF        | PDF document processing |

## 📁 Project Structure

```text
RAG Chatbot/
│
├── agent.py              # LangGraph support agent
├── app.py                # Streamlit application
├── ingestion.py          # Document ingestion and vectorization
├── rag.py                # RAG retrieval and response generation
├── tools.py              # Knowledge base tool
├── requirements.txt      # Python dependencies
├── .gitignore            # Files excluded from Git
│
└── data/                 # Local knowledge-base documents
```

> The original PDF knowledge base and generated vector database are intentionally excluded from the repository.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd RAG-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file:

```env
HF_TOKEN=your_huggingface_token
```

Do **not** commit `.env` to GitHub.

### 5. Install and run Ollama

Install Ollama and pull the model used by the application.

For example:

```bash
ollama pull qwen2.5:7b
```

Make sure Ollama is running before starting the application.

### 6. Add the knowledge-base document

Place your permitted PDF knowledge-base document inside:

```text
data/
```

### 7. Build the vector database

Run:

```bash
python ingestion.py
```

This creates the local vector store used by the RAG pipeline.

### 8. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔍 Key Features

### Retrieval-Augmented Generation

The system retrieves relevant information from a document knowledge base before generating an answer.

### Tool-Based Retrieval

The RAG functionality is exposed as a knowledge-base search tool that can be invoked by the LangGraph agent.

### Agent Orchestration

LangGraph manages the agent workflow and provides a structured state-based approach for connecting the LLM and tools.

### Local LLM

Ollama is used for local LLM inference, allowing the project to run without requiring a paid OpenAI API.

### Vector Search

ChromaDB provides persistent vector storage and similarity-based document retrieval.

## 📊 RAG Evaluation Considerations

The project can be evaluated using metrics such as:

* Retrieval Precision
* Retrieval Recall
* Context Relevance
* Faithfulness
* Answer Relevance
* Response Latency

These metrics can be incorporated into future iterations to monitor retrieval quality and overall agent performance.

## 🔮 Future Improvements

* Add automated RAG evaluation
* Add retrieval and response quality metrics
* Add source citations to responses
* Add conversation memory
* Add multiple knowledge-base documents
* Add document-level access control
* Add agent observability and tracing
* Add automated evaluation datasets
* Deploy the application to a cloud platform

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation
* Vector databases
* Embedding models
* Semantic search
* LLM integration
* Tool calling
* Agent orchestration
* LangGraph
* Local LLM deployment
* Document processing
