# chatbot.py
# Updated: Full conversation history + better follow-up support

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Please add your GOOGLE_API_KEY to .env file!")
    st.stop()

st.title("🧾 Tax Guru – Your ICAP CAF-2 Tax Assistant")
st.caption("Ask anything from Tax books. I remember our conversation, so feel free to ask follow-ups! 😊")

# Load vector DB
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 6})

retriever = load_db()

# Load prompt from file
try:
    with open("prompt_template.txt", "r", encoding="utf-8") as f:
        system_template = f.read()
except FileNotFoundError:
    st.error("prompt_template.txt not found!")
    st.stop()

# Initialize chat history FIRST (before chain definition)
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Optional welcome message
    welcome = """Welcome to Tax Guru! 👋I am your ICAP CAF-2 Tax Practices assistant, built exclusively using your official study material, question bank, and model papers (latest Tax Year 2026 rules).
            ✅ I answer ALL questions strictly from these PDFs – no external knowledge or guesses.
            ✅ For current exam preparation: My answers match the latest rules, rates, and suggested solutions.
            ✅ For old past papers: I use the core concepts and workings from the latest material. Note that tax rates, slabs, and some treatments may have changed over the years due to Finance Acts. Past paper solutions used the rates applicable in that specific tax year.
            Focus on understanding the concepts and workings – they remain highly relevant! 😊
            Let’s start – ask me anything!"""
                                            
    # st.session_state.messages.append({"role": "assistant", "content": welcome})
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# Function to format chat history for the LLM
def format_chat_history(messages):
    """Convert Streamlit messages to LangChain message format"""
    history = []
    # Get last 10 messages (excluding welcome message if it's the only one)
    recent_messages = messages[-10:] if len(messages) > 1 else []
    
    for msg in recent_messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

# Updated prompt with history support
# Include context in the system message
full_template = system_template + "\n\nUse the following context from the documents to answer the question:\n{context}"
prompt = ChatPromptTemplate.from_messages([
    ("system", full_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Using gemini-pro for stability (change if needed)
    google_api_key=google_api_key,
    temperature=0.4
)

# Chain with history and context
def create_chain_input(user_question):
    """Create input for the chain with proper context and history"""
    # Get context from retriever
    context_docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Format chat history (safely access session state)
    messages = st.session_state.get("messages", [])
    chat_history = format_chat_history(messages)
    
    return {
        "context": context,
        "question": user_question,
        "chat_history": chat_history
    }

chain = (
    RunnablePassthrough()
    | RunnableLambda(create_chain_input)
    | prompt
    | llm
    | StrOutputParser()
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a question or say hello..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke(user_input)
            except Exception as e:
                error_msg = str(e)
                if "messages" in error_msg and "attribute" in error_msg.lower():
                    # If messages not initialized, reinitialize and retry
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    response = f"❌ Error: {error_msg}\n\nPlease refresh the page and try again."
                elif "404" in error_msg or "NOT_FOUND" in error_msg:
                    response = "❌ Error: The Gemini model is not found. Please check the model name in the code (try 'gemini-pro' or 'gemini-1.5-pro-latest')."
                else:
                    response = f"❌ An error occurred: {error_msg}"
        st.markdown(response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})