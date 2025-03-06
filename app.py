# from langchain.vectorstores import Pinecone
from pinecone import Pinecone
from transformers import BertTokenizer, BertModel
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os

import streamlit as st
import time

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY" #
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY" #

chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

index_name = "llama-2"
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(index_name) # connect to index
print("Connected to index")

def get_augmented_prompt(query):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")

    encoded_input = tokenizer(query, return_tensors='pt')
    embeds = model(**encoded_input).pooler_output[0]

    s = index.query(
    namespace="",
    vector=embeds,
    top_k=1,
    include_values=False,
    include_metadata=True,
    )
    context = s['matches'][0]['metadata']['text']

    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {context}

    Query: {query}"""

    return augmented_prompt

def generate_response(chat, messages):
    
    response = chat(messages)
    print("response", response.content)

    for word in response.split():
        print("word", word)
        yield word + " "
        time.sleep(0.5)

st.title("RAG - Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    messages = [SystemMessage(content="You are a helpful assistant.")]
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    augmented_prompt= get_augmented_prompt(prompt)
    st.chat_message("user").markdown(augmented_prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": augmented_prompt})
    messages.append(HumanMessage(content=augmented_prompt))
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(chat, messages))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    messages.append(AIMessage(content=response))