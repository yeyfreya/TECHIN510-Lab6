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

# Display a title and description on the app
st.title("AI Writer Helper Chatbox")
st.write("""
This AI-powered chatbox helps you improve your writing by offering feedback on your text. 
Upload a document or input text directly, and the AI will provide suggestions on grammar, style, and content. 
Feel free to ask specific questions about your document or request general writing advice.
""")

# Initialize session state for chat messages if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Upload a document or enter text for revision to get started."}
    ]

# Chat input for text
user_input = st.text_input("Enter your text here for analysis and feedback:")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Send the user input to OpenAI API for processing
    try:
        response = client.Completion.create(
            model="text-davinci-003",  # Replace with your desired model
            prompt=user_input,
            temperature=0.7,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        ai_response = response.choices[0].text.strip()
    except Exception as e:
        ai_response = "Sorry, I couldn't process that request."
    
    st.session_state["messages"].append({"role": "assistant", "content": ai_response})


# File uploader widget to allow users to upload their documents
uploaded_file = st.file_uploader("Upload your document here", type=['pdf', 'txt'])

if uploaded_file:
    bytes_data = uploaded_file.read()
    
    if uploaded_file.type == "application/pdf":
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(bytes_data)
            reader = PDFReader()
            docs = reader.load_data(tmp.name)
            llm = LlamaOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                model="gpt-3.5-turbo",
                temperature=0.0
            )
            index = VectorStoreIndex.from_documents(docs)
        os.remove(tmp.name)
    elif uploaded_file.type == "text/plain":
        docs = [bytes_data.decode("utf-8")]

    # Example to display a simple message. Adjust according to your file processing and AI interaction logic.
    st.session_state["messages"].append({"role": "user", "content": "Document uploaded. Please ask your question."})

# Display the conversation
for message in st.session_state["messages"]:
    if message["role"] == "assistant":
        st.container().markdown(f"**AI**: {message['content']}")
    else:
        st.container().markdown(f"**You**: {message['content']}")

