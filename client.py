import streamlit as st
import pandas as pd
from io import StringIO
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import os

# Streamlit UI
st.title("🤖 Chat with DataBot powered by Charak Centers")
st.write("Upload a CSV file and ask me anything about the dataset!")

# Session state for API key, chat history, and file
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 1: Input OpenAI API Key
st.subheader("Step 1: Enter Your OpenAI API Key")
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if api_key:
    # Store the API key in session state
    st.session_state.api_key = api_key
    # Set the API key as an environment variable
    os.environ["OPENAI_API_KEY"] = api_key

# Initialize OpenAI LLM for LangChain if API key is provided
if st.session_state.api_key:
    llm = OpenAI(temperature=0.7, openai_api_key=st.session_state.api_key)

    # Updated prompt template
    prompt = PromptTemplate(
        input_variables=["combined_input"],
        template=(
            "You are a helpful and knowledgeable DataBot. The user has uploaded a dataset and is asking questions about it.\n\n"
            "{combined_input}\n\n"
            "Your response:"
        ),
    )

    # LangChain memory for conversation context
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Create LangChain chain
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # Step 2: File uploader
    st.subheader("Step 2: Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            # Read and summarize the dataset
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file = uploaded_file

            # Generate dataset summary
            dataset_summary = f"Columns: {list(df.columns)}\n\nSummary:\n{df.describe(include='all').to_string()}"
            st.session_state.dataset_summary = dataset_summary

            # Display the uploaded CSV file
            st.write("Here's a preview of your uploaded dataset:")
            st.dataframe(df)

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty. Please upload a valid CSV file.")
            st.session_state.uploaded_file = None
            st.session_state.dataset_summary = None

    # Step 3: Chat interface
    st.subheader("Step 3: Chat with DataBot")
    query = st.text_input("Type your question about the dataset and press Enter 👇")

    if query and st.session_state.dataset_summary:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Combine input variables
        dataset_summary = st.session_state.dataset_summary
        history = "\n".join(
            [f"User: {msg['content']}" if msg["role"] == "user" else f"DataBot: {msg['content']}" for msg in st.session_state.chat_history]
        )
        combined_input = f"Dataset summary:\n{dataset_summary}\n\nConversation history:\n{history}\n\nUser's question:\n{query}"

        # Use LangChain to process the query
        response = chain.run({"combined_input": combined_input})

        # Add bot's response to chat history
        st.session_state.chat_history.append({"role": "bot", "content": response})

    # Display chat history above the input box
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "bot":
            st.markdown(f"**DataBot:** {message['content']}")

    # Clear chat button (optional)
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.uploaded_file = None
        st.session_state.dataset_summary = None
        st.experimental_rerun()

    # If no file is uploaded
    if not st.session_state.dataset_summary:
        st.info("Please upload a dataset file first to start chatting.")
else:
    st.warning("Please enter your OpenAI API key to proceed.")




