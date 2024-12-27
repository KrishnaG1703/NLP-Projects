import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import chromadb
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()



def get_vectorstore_from_url(url):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = WebBaseLoader(url)
    document = loader.load()
    
    #split documents
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    document_texts = [chunk.page_content for chunk in document_chunks]

    vectorstore = FAISS.from_texts(document_texts, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index created and saved successfully.")
    return vectorstore

def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up information relevant to above conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    
def get_conversational_rag_chain(retrieval_chain):

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}")
    ])

    stuff_document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retrieval_chain, stuff_document_chain)

def get_response(user_query):
    #create an conversational chain
    retrieval_chain = get_context_retriever_chain(st.session_state.vector_store) 
    conversational_rag = get_conversational_rag_chain(retrieval_chain)  
    
    response = conversational_rag.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })
    
    return response['answer']


#app config
st.set_page_config(page_title="Chat with any website")
st.title("Chat with any website!")


with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    
if website_url is None or website_url == "":
    st.info("Please enter an Website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)


    #user input
    user_query = st.chat_input("Type your question here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        
        
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("ai"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.write(message.content)


