# chatbot.py
# Final Merged: Chat + Practice Paper Generator

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Please add your GOOGLE_API_KEY to .env file!")
    st.stop()

# App Title
st.title("🧾 Tax Guru – Your Complete ICAP CAF-2 Tax Assistant")
st.caption("Chat with me for explanations or switch to Practice tab to generate mock papers! 😊")

# Load DB
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
    return db

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 8})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Your working model
    google_api_key=google_api_key,
    temperature=0.4
)

# Load system prompt
try:
    with open("prompt_template.txt", "r", encoding="utf-8") as f:
        system_template = f.read()
except FileNotFoundError:
    st.error("prompt_template.txt not found!")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["💬 Chat with Tax Guru", "📝 Practice Paper Generator"])

# ====================== TAB 1: CHAT ======================
with tab1:
    # Chat prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chat_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: st.session_state.get("chat_history", [])
        }
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    # Initialize chat history and welcome message
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        welcome = """
Assalam o Alaikum! 👋  
I'm Tax Guru, your friendly ICAP CAF-2 Tax Practices tutor.  
I'm built using your official study material (TY 2026 rules). Ask me anything — theory, numericals, MCQs, or past papers! 😊
"""
        st.session_state.chat_history.append({"role": "assistant", "content": welcome})

    # Display all previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat input ALWAYS at the very bottom ---
    user_input = st.chat_input(
        "Ask a Tax question or say hello... 😊",
        key="persistent_chat_input"
    )

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_chain.invoke(user_input)
            st.markdown(response)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Rerun to scroll to bottom and keep input visible
        st.rerun()        
# ====================== TAB 2: PRACTICE PAPER ======================
with tab2:
    st.markdown("Generate **ICAP-style MCQs and numericals** with proper exam formatting. Attempt first, then reveal answers!")

    # Question generation prompt
    question_prompt = PromptTemplate.from_template("""
You are an expert ICAP CAF-2 Tax Practices examiner. Generate high-quality, exam-style questions.

User request: {user_request}

Instructions:
- Generate ONLY the requested number and type.
- MCQs: Number the question, full statement, then options on separate lines:
  a) 
  b) 
  c) 
  d) 
- Numericals: Full scenario + "Required:" clearly.
- Use latest TY 2026 rules from context.
- Do NOT provide answers.

Context:
{context}

Questions:
""")

    question_chain = (
        {"context": retriever, "user_request": RunnablePassthrough()}
        | question_prompt
        | llm
        | StrOutputParser()
    )

    # Answer prompt
    answer_prompt = PromptTemplate.from_template("""
Provide full ICAP-style suggested answers with step-by-step workings, section references, and final answers.

Questions:
{questions}

Suggested Answers:
""")

    answer_chain = answer_prompt | llm | StrOutputParser()

    request = st.text_input(
        "What do you want to practice?",
        placeholder="e.g., Generate 10 MCQs on Income from Salary, 3 numericals on Sales Tax"
    )

    if st.button("Generate Questions", key="gen_practice"):
        if request:
            with st.spinner("Generating questions..."):
                questions = question_chain.invoke(request)
            st.session_state.practice_questions = questions

            st.markdown("### 📄 Your Practice Paper")
            st.markdown(f"**Topic:** {request}")
            st.markdown("---")
            st.markdown(questions)
            st.success("Attempt them first! Then click below for answers. Good luck! 🚀")
        else:
            st.warning("Please enter your request!")

    if "practice_questions" in st.session_state:
        if st.button("Show Suggested Answers", key="show_ans"):
            with st.spinner("Generating full workings..."):
                answers = answer_chain.invoke({"questions": st.session_state.practice_questions})
            st.markdown("### ✅ Suggested Answers")
            st.markdown(answers)
