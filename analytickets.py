import os
import pandas as pd
import logging

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader 
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.indexes import VectorstoreIndexCreator

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

from langchain.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms.bedrock import Bedrock
import boto3

from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredExcelLoader   

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.callbacks import get_openai_callback

from langchain import PromptTemplate
from langchain import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain import ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.chains.question_answering import load_qa_chain

import streamlit as st
import re

from definitions import (
   
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_OPENAI,    
    LLM_MODEL_BEDROCK,
    LLM_MODEL_LOCAL,
    LLM_EMBEDDINGS_LOCAL,
    LLM_EMBEDDINGS_OPENAI,
    LLM_EMBEDDINGS_BEDROCK,
    STREAMLIT_DEBUG_MODE,
    LOCAL_MODEL_ID,
    LOCAL_MODEL_BASENAME,
)

def scrub_spi (input):

    #Mask IP Addresses
    ip_addr_regex = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
    scrubbed_response = re.sub(ip_addr_regex, ":blue[<a.b.c.d>]", input)

    #Mask Dates
    date_regex = re.compile("\d*\s*(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun| June|Jul|July|Aug|August|Sept|September|Oct|October|Nov|November|Dec|December)\s\d\d\d\d")
    scrubbed_response = re.sub(date_regex, " :blue[<Date Masked>]", scrubbed_response)

    #Mask Email Addresses
    email_regex = re.compile (r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
    scrubbed_response = re.sub(email_regex, " :blue[<Email Masked>]", scrubbed_response)

    return scrubbed_response


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename, resume_download=True)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        

os.environ["OPENAI_API_KEY"] = "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"

knowledgebase_index = "./ticket-index"

if STREAMLIT_DEBUG_MODE == 0:
    if 'working' not in st.session_state:
        st.session_state['working'] = 0

    if 'knowledgebase' not in st.session_state:
        st.session_state['knowledgebase'] = 0

if LLM_MODEL_OPENAI:
    model = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)
elif LLM_MODEL_LOCAL:
    model = ChatOllama(model="llama2",
                    verbose=True,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

elif LLM_MODEL_BEDROCK:
    boto3_bedrock = boto3.client(service_name='bedrock')
    model = Bedrock (model_id="amazon-titan-tg1-large", 
                     model_kwargs={
                         "maxTokenCount": 4096,
                         "stopSequences": [],
                         "temperature": 0,
                         "topP": 1
                     },
                     client=boto3_bedrock,
                    )
    
if STREAMLIT_DEBUG_MODE == 0:
    working = st.session_state['working']
else: 
    working = 0

if working == 0:
    if LLM_EMBEDDINGS_LOCAL == 1:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
        )
    
    elif LLM_EMBEDDINGS_OPENAI == 1:
        embeddings = OpenAIEmbeddings()
        
    elif LLM_EMBEDDINGS_BEDROCK == 1:
        embeddings = BedrockEmbeddings(client=boto3_bedrock)
        
    if STREAMLIT_DEBUG_MODE == 0:
        st.session_state['knowledgebase'] = FAISS.load_local(knowledgebase_index, embeddings)
    else:
        knowledgebase = FAISS.load_local(knowledgebase_index, embeddings)

# Prompt templates
Prompt_template = PromptTemplate(
    input_variables = ["context", "query"],
    template = """
    Use the following pieces of context to answer the question at the end. 
    Use numbered lists to structure your answer where appropriate.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    The IT environment that applies to the provided context is different from the IT environment 
    that applies to the question at the end, so please word your answer accordingly. 
    In your answer, whenever you refer to specifics detailed in the supplied context or if you want to use
    the word "context", word it as "for a past incident described in the supplied knowledgebase". Thus, instead
    of writing "it is mentioned that", write "in the case of a past incident in the knowledgebase,
    it is mentioned that". Instead of saying 
    "according to investigation by" write "according to investigation of a past incident by". 
    Whenever you want to write "in the past incident", instead write "in a past incident".
    Whenever you want to write "based on the provided context", write "based on the provided knowledgebase".

    The following is the context: {context} 

    Question: {query}. Can you determine and resolve the cause of this problem based on the provided knowledgebase?

    Helpful Answer:
    """
)

Prompt_template_SPI = PromptTemplate(
    input_variables = ["text_with_spi"],
    template = """

    Mask IPv4 addresses and date information in the following text while keeping all other information intact. 
    An IPv4 address is a 32-bit numerical label, expressed in dotted-decimal format (e.g., 192.168.1.1) 
    used for device identification on IP networks, distinct from regular plain numbers.
   
    Text: {text_with_spi}

    """
)

Prompt_template_native = PromptTemplate(
    input_variables = ["query"],
    template = """
    The following question relates a problem that I have observed in my IT environment: {query}
    Please suggest resolution steps.
    """
)
    
chain = load_qa_chain(llm=model, chain_type="stuff")

if STREAMLIT_DEBUG_MODE == 0:
    st.title("Snag Sherpa")
    st.subheader("_An AI Bot that studies history to help you solve IT problems_")

    print ("===AI Assistant for Incident Resolution ===")

    query = st.text_input ("""Please describe the problem in your IT environment that you want to resolve. 
    I will attempt to advise you based on (and only based on) past ticketing data in my knowledgebase. I will try and mask SPI.    
    :violet[**Usage Examples**]: :orange[The BR.NET application in my environment is down.]
    :green[How can I resolve the problem of not being able to connect to Ansible jump servers?]
    :blue[How do I resolve the problem that RDP services are not working even though the server is reachable over the network?]
    :green[How do I resolve database backup failures?]
    :orange[How do i resolve storage degradation and disk error problems?]
    :blue[How do I resolve the problem of servers rebooting unexpectedly?]
    :green[How do I recover if an AIX Cluster is down?]
    :orange[SAP Basis servers are inaccessible.]
    :blue[How do I resolve the problem of FTP servers hanging?]
    :green[How do i recover a machine with a lost root password?]""", 
    key="input")
else:
    #query = "How do I recover if my BR.NET application is down?"
    query = "How can I resolve the problem of not being able to connect to Ansible jump servers?"
    #query = "How do I resolve database backup failures?"
    #query = "How do I resolve the problem that RDP services are not working even though the server is reachable over the network?"
    #query = "How do i resolve storage degradation and disk error problems?"
    #query = "How do I resolve the problem of servers rebooting unexpectedly?"
    #query = "How do I recover if an AIX Cluster is down?"

    #query = "SAP Basis servers are inaccessible."
    #query = "How do I resolve the problem of FTP servers hanging?"
    #query = "How do i recover a machine with a lost root password?"

if query:

    if STREAMLIT_DEBUG_MODE == 0:
        st.session_state['working'] = 1
        knowledgebase = st.session_state['knowledgebase']

    docs = knowledgebase.similarity_search(query)
    print ("Number of matches =", len(docs))

    # Scrub SPI
    for x in docs:
        x.page_content = scrub_spi (x.page_content)

    if STREAMLIT_DEBUG_MODE == 0:
        st.write (":violet[I have located the top", len(docs), """semantic matches for your query within my knowledgebase. 
                    Please wait as I figure out resolution steps for you. 
                    I will also point you to past incidents of interest at the end. This might take a minute..] :sunglasses:""")

    # Interpret from past ticket data
    chain = LLMChain(llm=model, prompt=Prompt_template)

    with get_openai_callback() as cb:
        response = chain.run(context=docs , query=query)
        print (cb)

    print(response)
    

    if STREAMLIT_DEBUG_MODE == 0:
        st.markdown (response)

        print ("I also suggest that you look at the following past problem tickets:")
        st.write ("I also suggest that you look at the following past problem tickets:")

    tick_list=[]

    num=0
    while (num < len(docs)):
        tick_full = docs[num].metadata['source']
        tokens = tick_full.split('/')
        tick = tokens[len(tokens)-1]

        if tick not in tick_list:
            new_tokens = tick.split('.')
            new_tick = new_tokens[0]
            tick_list.append(new_tick)
            tick_components = new_tick.split('_')

            if len(tick_components) >= 3:
                print ("Account: ", tick_components[0], ", Resolution Date: ", 
                tick_components[1], ", Incident #: ", tick_components[2])
                if STREAMLIT_DEBUG_MODE == 0:
                    st.write (":blue[Account:] ", tick_components[0], ", :green[Resolution Date:] ", 
                tick_components[1], ", :orange[Incident #:] ", tick_components[2])
            print (tick_full)

        num+=1

if 0:
    
    chain = LLMChain(llm=model, prompt=Prompt_template_SPI)

    with get_openai_callback() as cb:
        scrubbed_response = chain.run(text_with_spi=response)
        print (cb)
    print (scrubbed_response)
    st.markdown (scrubbed_response)