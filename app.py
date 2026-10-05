import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Support Agent",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🤖 AI Support Agent")

st.write(
    "Welcome! Ask a question about the support knowledge base."
)

st.divider()

# ---------------------------------------------------------
# Test that Streamlit is working
# ---------------------------------------------------------

st.success("Streamlit application is running successfully.")

# ---------------------------------------------------------
# Import Agent
# ---------------------------------------------------------

try:

    from agent import support_agent

    st.success("LangGraph agent loaded successfully.")

except Exception as e:

    st.error("Could not load the LangGraph agent.")

    st.exception(e)

    st.stop()


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question..."
)


# ---------------------------------------------------------
# Run Agent
# ---------------------------------------------------------

if question:

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)

    # Save user message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Run LangGraph
    with st.chat_message("assistant"):

        with st.spinner("Agent is thinking..."):

            try:

                result = support_agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": question
                            }
                        ]
                    }
                )

                # Get final response
                final_message = result["messages"][-1]

                answer = final_message.content

                st.markdown(answer)

                # Save response
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error("Agent execution failed.")

                st.exception(e)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("AI Support Agent")

    st.write(
        """
        This application demonstrates:

        • RAG  
        • LangChain  
        • LangGraph  
        • ChromaDB  
        • Ollama 
        • Hugging Face Embeddings
        """
    )

    st.divider()

    st.subheader("Architecture")

    st.code(
        """
User
  ↓
LangGraph Agent
  ↓
Tool Decision
  ↓
RAG Tool
  ↓
ChromaDB
  ↓
Relevant Documents
  ↓
Ollama LLM
  ↓
Answer
        """
    )

    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()
