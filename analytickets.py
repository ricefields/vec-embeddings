import os
import pandas as pd

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

#os.environ["OPENAI_API_KEY"] = "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"

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large" 

LLM_MODEL_OPENAI = 1
LLM_MODEL_BEDROCK = 0

knowledgebase_index = "./ticket-index"

if 'working' not in st.session_state:
    st.session_state['working'] = 0

if 'knowledgebase' not in st.session_state:
    st.session_state['knowledgebase'] = 0

if LLM_MODEL_OPENAI:
    model = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)
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
    
if st.session_state['working'] == 0:
    if LLM_MODEL_OPENAI:

        #embeddings = HuggingFaceInstructEmbeddings(
        #    model_name=EMBEDDING_MODEL_NAME,
        #    #model_kwargs={"device": "mps"},
        #)

        embeddings = OpenAIEmbeddings()
        
    elif LLM_MODEL_BEDROCK:
        embeddings = BedrockEmbeddings(client=boto3_bedrock)

    st.session_state['knowledgebase'] = FAISS.load_local(knowledgebase_index, embeddings)

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

    Question: {query}.Can you determine and resolve the cause of this problem based on the provided knowledgebase?

    Helpful Answer:
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

#query = "How do I recover if my BR.NET application is down?"
#query = "How can I resolve the problem of not being able to connect to Ansible jump servers?"
#query = "How do I resolve database backup failures?"
#query = "How do I resolve the problem that RDP services are not working even though the server is reachable over the network?"
#query = "How do i resolve storage degradation and disk error problems?"
#query = "How do I resolve the problem of servers rebooting unexpectedly?"
#query = "How do I recover if an AIX Cluster is down?"

#query = "SAP Basis servers are inaccessible."
#query = "How do I resolve the problem of FTP servers hanging?"
#query = "How do i recover a machine with a lost root password?"

st.title("Snag Sherpa")
st.subheader("_An AI Bot that studies history to help you solve IT problems_")

print ("===AI Assistant for Incident Resolution ===")

#with col1:
query = st.text_input ("""Please describe the problem in your IT environment that you want to resolve. 
I will attempt to advise you based on (and only based on) past ticketing data in my knowledgebase.      
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

if query:
    st.session_state['working'] = 1

    knowledgebase = st.session_state['knowledgebase']
    docs = knowledgebase.similarity_search(query)
    print ("Number of matches =", len(docs))
    st.write (":violet[I have located the top", len(docs), """semantic matches for your query within my knowledgebase. 
                Please wait as I figure out resolution steps for you. 
                I will also point you to past incidents of interest at the end. This might take a minute..] :sunglasses:""")

    # Interpret from past ticket data
    chain = LLMChain(llm=model, prompt=Prompt_template)

    with get_openai_callback() as cb:
        response = chain.run(context=docs , query=query)
        print (cb)

    print(response)
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
                st.write (":blue[Account:] ", tick_components[0], ", :green[Resolution Date:] ", 
                tick_components[1], ", :orange[Incident #:] ", tick_components[2])
            print (tick_full)

        num+=1

if 0:
    # Use OpenAI as consultant rather than analyzing past ticket data
    chain = LLMChain(llm=model, prompt=Prompt_template_native)

    with get_openai_callback() as cb:
        response = chain.run(query=query)
        print (cb)
    print (response)