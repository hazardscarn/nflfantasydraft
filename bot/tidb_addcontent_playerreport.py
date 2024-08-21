import os
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import TiDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy import create_engine, text
import glob
import pandas as pd
import numpy as np
from langchain.docstore.document import Document
import re

# Load config
with open('static/config.yml', 'r') as file:
    config = yaml.safe_load(file)

PLAYER_INFO_TABLE_NAME = config['vectordb']['playerreport']
EMBEDDING_MODEL = config['EMBEDDING_MODEL']

# Load environment variables
load_dotenv()
tidb_connection_string = os.getenv('TIDB_CONNECTION_URL')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Google embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

def load_csv_files(directory):
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    if not all_files:
        raise Exception(f"No CSV files found in directory {directory}")
    
    df_from_each_file = (pd.read_csv(f, keep_default_na=False) for f in all_files)
    
    try:
        concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    except ValueError as e:
        raise Exception("Unable to concatenate CSV files. Check if the files are not empty.") from e
    
    concatenated_df.reset_index(drop=True, inplace=True)
    
    # Convert numerical columns to appropriate types
    numerical_columns = ['non_ppr_projection', 'ppr_projection', 'bye_week', 'ADP']
    for col in numerical_columns:
        concatenated_df[col] = pd.to_numeric(concatenated_df[col], errors='coerce')
    
    # Handle 'rookie' column
    concatenated_df['rookie'] = concatenated_df['rookie'].fillna('No')
    concatenated_df['rookie'] = np.where(concatenated_df['rookie'] == '', 'No', 'Yes')
    
    # Clean the Outlook column
    def clean_outlook(row):
        player_name = row['player']
        outlook = row['Outlook']
        
        # Split player name into parts
        name_parts = player_name.split()
        
        # Create a regex pattern to match any part of the name
        pattern = '|'.join(re.escape(part) for part in name_parts)
        
        # Remove player name parts from the outlook
        cleaned = re.sub(pattern, '', outlook, flags=re.IGNORECASE)
        
        # Remove any extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned

    concatenated_df['Outlook'] = concatenated_df.apply(clean_outlook, axis=1)
    
    return concatenated_df

def create_documents_from_dataframe(df):
    documents = []
    
    for _, row in df.iterrows():
        metadata = row.drop('Outlook').to_dict()
        
        # Ensure numerical values are properly formatted for metadata
        for key, value in metadata.items():
            if pd.isna(value):
                metadata[key] = None
            elif isinstance(value, (int, float)):
                metadata[key] = float(value)  # Convert all numbers to float for consistency
            elif key == 'rookie':
                metadata[key] = 'Yes' if value == 'Yes' else 'No'
        
        doc = Document(page_content=row['Outlook'], metadata=metadata)
        documents.append(doc)
    
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
            SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(meta, '$.player')) as player
            FROM {table_name}
            WHERE JSON_EXTRACT(meta, '$.player') IS NOT NULL
        """)

        with engine.connect() as connection:
            result = connection.execute(query)
            existing_players = set(row[0] for row in result)

        new_docs = [doc for doc in docs if doc.metadata['player'] not in existing_players]
        
        if new_docs:
            existing_db.add_documents(new_docs)
            print(f"Added {len(new_docs)} new player outlooks to the vector store.")
        else:
            print("No new player outlooks to add.")
        
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
        print(f"Created new vector store with {len(docs)} player outlooks.")
    
    return db

def main():
    # Load CSV files
    df = load_csv_files("bot/documents/player_outlook")
    print(f"Loaded {df.shape[0]} player records")

    # Create documents from dataframe without chunking
    documents = create_documents_from_dataframe(df)
    print(f"Created {len(documents)} player outlook documents")

    # Create or update vector store
    db = create_or_update_vector_store(documents, embeddings, tidb_connection_string, PLAYER_INFO_TABLE_NAME)

    # Example query with metadata filtering
    query = "J K Dobbins"
    docs_with_score = db.similarity_search_with_score(query, k=3)

    for doc, score in docs_with_score:
        print("-" * 80)
        print("Score: ", score)
        print("Player:", doc.metadata['player'])
        print("Position:", doc.metadata['pos'])
        print("Team:", doc.metadata['team'])
        print("ADP:", doc.metadata['ADP'])
        print("PPR Projection:", doc.metadata['ppr_projection'])
        print("Rookie:", doc.metadata['rookie'])
        print("Cleaned Outlook:", doc.page_content[:200] + "...")  # Print first 200 characters of cleaned outlook
        print("-" * 80)

if __name__ == "__main__":
    main()