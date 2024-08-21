import os
import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import TiDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()
tidb_connection_string = os.getenv('TIDB_CONNECTION_URL')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Google embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

TABLE_NAME = "ffyoutube"

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_text_file(file_path, url, title, channel_name):
    loader = TextLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata['source'] = 'txt'
        doc.metadata['title'] = title
        doc.metadata['url'] = url
        doc.metadata['channel_name'] = channel_name
    return documents

def process_articles(config):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for channel_name, videos in config['youtube'].items():
        for video_key, video_info in videos.items():
            file_path = os.path.join(config['output_path'], f"{channel_name}_{video_key}.txt")
            url = video_info['url']
            title = video_info['title']
            
            if os.path.exists(file_path):
                try:
                    txt_docs = load_text_file(file_path, url, title, channel_name)
                    split_txt_docs = text_splitter.split_documents(txt_docs)
                    documents.extend(split_txt_docs)
                    print(f"Loaded and processed text file: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error loading text file {file_path}: {e}")
            else:
                print(f"File not found: {file_path}")

    return documents

def create_or_update_vector_store(docs, embeddings, connection_string, table_name):
    try:
        existing_db = TiDBVectorStore.from_existing_vector_table(
            embedding=embeddings,
            connection_string=connection_string,
            table_name=table_name
        )

        engine = create_engine(connection_string)
        query = text(f"""
            SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(meta, '$.url')) as url 
            FROM {table_name}
            WHERE JSON_EXTRACT(meta, '$.url') IS NOT NULL
        """)

        with engine.connect() as connection:
            result = connection.execute(query)
            existing_urls = set(row[0] for row in result)

        new_docs = [doc for doc in docs if doc.metadata['url'] not in existing_urls]
        
        if new_docs:
            existing_db.add_documents(new_docs)
            print(f"Added {len(new_docs)} new document chunks to the vector store.")
        else:
            print("No new documents to add.")
        
        db = existing_db

    except Exception as e:
        print(f"Creating new vector store: {str(e)}")
        db = TiDBVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            table_name=table_name,
            connection_string=connection_string,
            distance_strategy="cosine"
        )
        print(f"Created new vector store with {len(docs)} document chunks.")
    
    return db

def main():
    config_path = 'bot//youtube_source.yml'  
    config = load_config(config_path)
    all_docs = process_articles(config)
    
    db = create_or_update_vector_store(all_docs, embeddings, tidb_connection_string, TABLE_NAME)

    # Example query
    query = "What are the top fantasy football sleepers for 2024?"
    docs_with_score = db.similarity_search_with_score(query, k=3)

    for doc, score in docs_with_score:
        print("-" * 80)
        print("Score: ", score)
        print("Channel:", doc.metadata['channel_name'])
        print("Title:", doc.metadata['title'])
        print("URL:", doc.metadata['url'])
        print(doc.page_content[:200] + "...")  # Print first 200 characters
        print("-" * 80)

if __name__ == "__main__":
    main()