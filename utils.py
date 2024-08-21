from flask import Flask, render_template, request, jsonify, Response
from models.draft_recommendation import FantasyFootballDraftAssistant
import logging
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import TiDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import json
import yaml
from NFLFantasyQA import NFLFantasyQA
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)



# Create retriever and chatbot chain
def create_retriever(embeddings,tidb_connection_string, ARTICLE_TABLE_NAME,search_kwargs=None):
    #embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_api_key)
    vector_store = TiDBVectorStore.from_existing_vector_table(
        embedding=embeddings,
        connection_string=tidb_connection_string,
        table_name=ARTICLE_TABLE_NAME
    )
    if search_kwargs is None:
        search_kwargs = {"k": 5}
    return vector_store.as_retriever(search_kwargs=search_kwargs)



def create_chatbot(retriever,google_api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001",
        temperature=0.3,
        top_p=0.95,
        google_api_key=google_api_key
    )

    prompt_template = """You are an NFL Fantasy Football expert.
    You are provided with the content from famous fantasy football channels on YouTube.
    You should answer the user's questions based on the context provided from the YouTube videos.

    Use the following pieces of context to answer the users question.
    Context: {context}
    Question: {question}
    

    
    - Always try to answer the user's question based on the context provided from the youtube channel transcript.
    
    **The output should be created in a markdown format.** 
    At the end of your response, provide a list of references used to create the answer, formatted as:
    References:
    1. [Title 1](URL 1)
    2. [Title 2](URL 2)
    ...
    """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | llm
    )

    return chain


def create_youtube_retriever(embeddings,tidb_connection_string,YOUTUBE_TABLE_NAME,search_kwargs=None, channels=None):
    # embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_api_key)
    vector_store = TiDBVectorStore.from_existing_vector_table(
        embedding=embeddings,
        connection_string=tidb_connection_string,
        table_name=YOUTUBE_TABLE_NAME
    )
    if search_kwargs is None:
        search_kwargs = {"k": 5}
    
    if channels:
        filter_func = lambda doc: any(channel in doc.metadata['channel_name'] for channel in channels)
        return vector_store.as_retriever(search_kwargs=search_kwargs, filter=filter_func)
    else:
        return vector_store.as_retriever(search_kwargs=search_kwargs)



def create_nfl_fantasy_chatbot(question,nfl_fantasy_qa):
    try:
        logger.info(f"Processing question: {question}")
        response = nfl_fantasy_qa.get_answer(question)
        logger.debug(f"Raw response: {response}")
        content, references = split_response(response)
        
        def generate():
            for chunk in content.split('\n'):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            if references:
                yield f"data: {json.dumps({'references': references})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return generate()
    except Exception as e:
        logger.exception("Error in NFL Fantasy chatbot response generation")
        return iter([f"data: {json.dumps({'error': str(e)})}\n\n"])
    

def split_response(response):
    parts = response.split("## References")
    content = parts[0].strip()
    if len(parts) > 1:
        references = parts[1].strip()
        if references and not references.startswith("No specific sources cited"):
            references = "Sources: " + references
        else:
            references = ""  # Don't include generic reference messages
    else:
        references = ""
    return content, references