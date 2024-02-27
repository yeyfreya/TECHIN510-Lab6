from tempfile import NamedTemporaryFile
import os

from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.readers.file import PDFReader

# Load .env file to use environment variables
load_dotenv()

# Initialize the OpenAI client with API key and base URL from .env
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="AI Writer Helper",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="auto",
)

# Display a title on the app
st.title("AI Writer Helper Chatbox")

# Add a description about what the chatbox does
st.write("""
This AI-powered chatbox helps you improve your writing by offering feedback on your text. 
Upload a document or input text directly, and the AI will provide suggestions on grammar, style, and content. 
Feel free to ask specific questions about your document or request general writing advice.
""")

# Initialize session state for chat messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [
            {"role": "assistant", "content": "Upload a document or enter text for revision to get started."}
        ]

# Display a title on the app
st.title("AI Writer Helper")

# File uploader widget to allow users to upload their documents
uploaded_file = st.file_uploader("Upload your document here", type=['pdf', 'txt'])

# Process the uploaded file
if uploaded_file:
    # Read the content of the file
    bytes_data = uploaded_file.read()
    
    # Depending on the file type, process differently
    if uploaded_file.type == "application/pdf":
        # Process PDF file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(bytes_data)
            reader = PDFReader()
            docs = reader.load_data(tmp.name)
            llm = LlamaOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                model="gpt-3.5-turbo",
                temperature=0.0,
                system_prompt="You are an expert on the content of the document, provide detailed answers to the questions. Use the document to support your answers.",
            )
            index = VectorStoreIndex.from_documents(docs)
        os.remove(tmp.name)  # Remove the temporary file
    elif uploaded_file.type == "text/plain":
        # For text files, treat the content as a single document
        docs = [bytes_data.decode("utf-8")]

    # Display the chat input box and process the conversation
    if prompt := st.chat_input("Enter your text for revision or ask a question about the uploaded document:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # For PDF processing with LlamaIndex
        if uploaded_file.type == "application/pdf":
            if "chat_engine" not in st.session_state:
                st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False, llm=llm)
            
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.stream_chat(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
        else:
            # For text revision, use OpenAI directly
            response = client.completions.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=1024,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            st.session_state.messages.append({"role": "assistant", "content": response.choices[0].text})

# Display the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
