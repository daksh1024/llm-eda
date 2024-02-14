import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import markdown


from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
#from langchain.llms import OpenAI
import pathlib
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from more_itertools import batched

from llama_index.query_engine import PandasQueryEngine

import summarize
import prompts
import tempfile

load_dotenv()

OpenAI.api_key = os.getenv('OPENAI_API_KEY_ANALYTICS')
client = OpenAI()
CHROMA_PATH = "csv_review_embeddings"
chromadb_client = chromadb.Client()
#chromadb_client.heartbeat()

#langchain
def create_agent(file):
    agent = create_csv_agent(
        OpenAI(temperature=0),
        file,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )    
    
    return agent

#llama-index
def create_query_engine(df):
    query_engine = PandasQueryEngine(df=df, verbose=False)
    
    return query_engine

def init_chromadb():
    client = chromadb.PersistentClient(path="/chromabd")
    vector_collection = client.create_collection("csv_vector_db")
    return vector_collection

def convert_csv_to_vector_dict(df): 
    prompt = prompts.csv_to_vectordb_prompt.format(df.columns.to_list(), df.iloc[0].to_list())

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",  # or other available engines
    messages=[
        {"role": "system", "content": prompt}
        ]
        )
    
    sentences = []
    for index, row in df.iterrows():
        sentence = f"{completion.choices[0].message.content}"
        sentences.append(sentence)

    # Create ids, documents, and metadatas data in the format chromadb expects
    ids = [f"row{i}" for i in range(df.shape[0])]
    metadatas = df.iloc[0].to_list()

    return {"ids": ids, "documents": sentences, "metadatas": metadatas}

def get_query_engine_response(query_engine, prompt):
    
    #sample prompt
    prompt = '''
    Which quadrant had the most incidents?
    '''
    return query_engine.query(prompt)

def add_to_chroma_collection(
    chroma_path: pathlib.Path,
    collection_name: str,
    embedding_func_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine"):

    """Create a ChromaDB collection"""

    print(chroma_path)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OpenAI.api_key, model_name="text-embedding-ada-002")
    collection = chromadb_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

    document_indices = list(range(len(documents)))

    #using batch size 150, max allowed size is 166
    try:
        for batch in batched(document_indices, 150):
            start_idx = batch[0]
            end_idx = batch[-1]

            collection.add(
                ids=ids[start_idx:end_idx],
                documents=documents[start_idx:end_idx]
                #metadatas=metadatas[start_idx:end_idx],
            )
        return f"Saved to collection {collection_name} :)"
    except:
        return "Something went wrong :("
    

def main():
    st.set_page_config(page_title="Automated Data Analysis")
    st.header("LLM powered data analysis")
    menu = st.sidebar.selectbox("Choose an option", ["Summarize", "Ask Query"])

    if menu == "Summarize":
        st.subheader("Summarization of your data")
        file_upload = st.file_uploader("Upload your file (csv, excel)", type=["csv", "xlsx"])
        if file_upload is not None:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, file_upload.name)
            file_extension = path.split('.')[-1]
            file_name = path.split('.')[0].split('/')[-1]
            if file_extension == "csv":
                df = pd.read_csv(file_upload)
            elif file_extension == "xlsx":
                df = pd.read_excel(file_upload)
          
                
            #crashes for very large df size (maybe needs gpu to speedup)
            #collection_length = df.shape[0]
            collection_length = 1000
            chroma_csv_dict = convert_csv_to_vector_dict(df.iloc[0:collection_length])
            #print(chroma_csv_dict[-10])

            DATA_PATH = "data/archive/*"
            CHROMA_PATH = "csv_review_embeddings"
            EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
            COLLECTION_NAME = file_name


            add_collection_status = add_to_chroma_collection(
                CHROMA_PATH,
                COLLECTION_NAME,
                EMBEDDING_FUNC_NAME,
                chroma_csv_dict["ids"],
                chroma_csv_dict["documents"],
                chroma_csv_dict["metadatas"]
            )
            st.write(add_collection_status)





    elif menu == "Ask Query":
        st.subheader("Query your data")
        file_uploader = st.file_uploader("Upload your file", type="csv")
        if file_uploader is not None:
            path_to_save = "filename2.csv"
            with open(path_to_save, "wb") as f:
                f.write(file_uploader.getvalue())
            
            st.write("Answer goes here")



if __name__ == "__main__":
    main()