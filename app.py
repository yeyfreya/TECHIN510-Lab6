from tempfile import NamedTemporaryFile
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI as OpenAIClient
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.readers.file import PDFReader

# Load .env file to use environment variables
load_dotenv()

# Initialize the OpenAI client with API key and base URL from .env
client = OpenAIClient(
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

# Initialize session state for chat messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about your document!"}]

# Display a title on the app
st.title("AI Writer Helper")

# File uploader widget to allow users to upload their documents
uploaded_file = st.file_uploader("Upload your document here", type=['pdf', 'txt'])

if uploaded_file:
    bytes_data = uploaded_file.read()
    if uploaded_file.type == "application/pdf":
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(bytes_data)
            with st.spinner("Processing document, please wait..."):
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
                if "chat_engine" not in st.session_state:
                    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False, llm=llm)
            os.remove(tmp.name)  # Clean up the temporary file
    elif uploaded_file.type == "text/plain":
        docs = [bytes_data.decode("utf-8")]

if prompt := st.chat_input("Enter your text for revision or ask a question about the uploaded document:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if uploaded_file and uploaded_file.type == "application/pdf":
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(prompt)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
    else:
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
