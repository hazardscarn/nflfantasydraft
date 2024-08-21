import yaml
import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import YoutubeLoader

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def clean_document(llm, content):
    prompt = ChatPromptTemplate.from_template("""
    You are an expert NFL fantasy AI assistant tasked with cleaning and  organize the content without losing any information for a RAG (Retrieval-Augmented Generation) system.
    You are an expert in reading large articles about NFL fantasy football and extracting the information from it.
    There is no personal information in the content and do not confuse that there is personal information in the content.
                                              
    Your task is to read articles about NFL fantasy football and clean and organize the content without losing any information.
    This content created will be used for a RAG system to provide information about NFL fantasy football. 
    If there are tables, lists, or any other structured content, you should clean and organize it in a way that is easy to read and understand.                                             
    The cleaned content should be informative and suitable for retrieval.
    You should not add any information that is not present in the content.
    
    Please process the following content:
    
    {content}
    
    Clean and summarize the content, ensuring that:
        1. All relevant information to NFL Fantasy football or drafting strategy is retained.
        2. The text is well-structured and easy to understand.
        3. None of the key information related to players, draft strategy or NFL fantasy is lost in the cleaning process.
        3. If there are any tables, lists, or structured content, clean and organize it in a way that is easy to read and understand.
        4. Main objective is to clean and organize the content without losing any information.
        5. You may remove unnecessary details, but make sure to retain all key information.
        6. Remove Terms of Service, Privacy Policy, or any other irrelevant information.
        7. Make sure that when you are summarizing content for a player, you add their name in the summary. Do not mistake player name with analyst name.
        8. If there is any additional information about the player like their position,depth,projection etc. , make sure to include it in the summary.
        9. If the content is about draft strategy explain the strategy in detail. Do not lose any information about here.
        10. Ignore URLS, page numbers, link to other articles, external tools, or any other irrelevant information.
        11. Make sure rankings, projections, tables etc are all retained and formatted properly. These are very important for the user.

    Your job is just to clean and summarize the content. Do not add any additional information or comments or notes that is not present in the content.                                              
                                              
    NOTE: 
        -  It is very important to not lose any fantasy football-related information in the cleaning  and summarization process.
        -  DO NOT add headings or titles to the content summary.
        -  DO NOT add anything other than cleaned and summarized content. 
        -  DONOT add text like "Here is the summary of the content".
        -  DO NOT add any information that is not present in the content. Do not add any notes or comments unless it is part of the content.
        -  DO NOT add any conclusion or opinion to the content unless it is part of the content.
        -  DO NOT say I cannot generate the requested content as I am unable to access external sources or extract information from the context.                                            
        -  DO NOT say  anything like this I am unable to provide the cleaned and summarized content requested, as I am unable to access external sources or provide personalized opinions.
        -  DO NOT create some sample content or provide any other information that is not present in the content.
        -  If the whole content is irrelevant or not related to NFL fantasy football, you can skip the content and return an empty string.
        -  It is very important that you don't add any information that is not present in the content.

    Provide the generated content below:
    """)

    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"content": content})

def process_youtube_videos(config_file):
    config = load_yaml(config_file)
    llm = Ollama(model="gemma2:2b",temperature=0.1)
    output_path = config['output_path']
    os.makedirs(output_path, exist_ok=True)

    # Initial splitter for processing
    process_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)

    for channel_name, videos in config.get('youtube', {}).items():
        for video_key, video_info in videos.items():
            url = video_info['url']
            title = video_info['title']
            try:
                logging.info(f"Loading YouTube video: {title} from channel: {channel_name}")
                loader = YoutubeLoader.from_youtube_url(url)
                video_content = loader.load()[0].page_content
                
                chunks = process_splitter.split_text(video_content)
                cleaned_chunks = []
                
                for i, chunk in enumerate(chunks, 1):
                    logging.info(f"Cleaning chunk {i}/{len(chunks)} of video: {title}")
                    print(f"Cleaning chunk {i}/{len(chunks)} of video: {title}")
                    #print(f"chunk: {chunk}")
                    cleaned_chunk = clean_document(llm, chunk)
                    #print(f"cleaned_chunk: {cleaned_chunk}")
                    cleaned_chunks.append(cleaned_chunk)
                
                # Save cleaned content to a txt file
                output_file_name = f"{channel_name}_{video_key}.txt"
                output_file_path = os.path.join(output_path, output_file_name)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"Channel: {channel_name}\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"Date: {video_info['date']}\n\n")
                    f.write("\n\n".join(cleaned_chunks))
                
                logging.info(f"Completed processing and saved YouTube video: {title} to {output_file_path}")
            except Exception as e:
                logging.error(f"Error processing YouTube video {video_key} from channel {channel_name}: {e}")

    logging.info("All YouTube videos processed and saved.")

if __name__ == "__main__":
    config_file = "bot//youtube_source.yml"
    logging.info(f"Starting YouTube video processing with config file: {config_file}")
    process_youtube_videos(config_file)
    logging.info("YouTube video processing completed")
