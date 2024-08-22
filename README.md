# NFL Fantasy Draft Assistant

## Overview

The NFL Fantasy Draft Assistant is a sophisticated application designed to help fantasy football enthusiasts make informed decisions during their draft. Leveraging advanced technologies such as TiDB Vector Database and LangGraph, this app provides real-time draft recommendations, player insights, and expert analysis.

## Key Features

1. **Draft Chatbot (LangGraph Agent)**: 
   - Powered by TiDB for efficient vector storage and retrieval
   - Utilizes Retrieval-Augmented Generation (RAG) for context-rich responses
   - Incorporates a web search agent for up-to-date information

2. **YouTube Expert Chatbot**:
   - Allows channel-specific searches using TiDB's flexible filtering
   - Performs RAG on vectorized YouTube video transcripts
   - Delivers targeted insights from chosen fantasy football experts

3. **Player Similarity Search**:
   - Utilizes TiDB's fast similarity search capabilities
   - Finds and compares players with similar attributes
   - Helps in discovering alternatives during drafts or for team management

4. **Draft Recommendation Engine**:
   - Provides AI-powered recommendations for player selections
   - Considers factors such as ADP, team needs, and player projections

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/hazardscarn/nflfantasydraft.git
   cd nflfantasydraft
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add the following variables:
     ```
     TIDB_CONNECTION_URL=your_tidb_connection_string
     GOOGLE_API_KEY=your_google_api_key
     ```

4. Ensure you have access to a TiDB cluster with the necessary vector tables set up.

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:10000` (or the port specified in your environment).

3. Use the various features of the app:
   - Draft Assistant: Get real-time advice during your fantasy draft
   - YouTube Expert: Ask questions and get insights from popular fantasy football YouTube channels
   - Player Similarity: Find players with similar attributes and projections

4. If you want to try out the aplication deployed on render:
    
    https://nflfantasydraft.onrender.com/

## Contributing

We welcome contributions to improve the NFL Fantasy Draft Assistant. Please feel free to submit issues or pull requests.

## Contact Information

For any queries or suggestions, please contact:

- Email: davidacad10@gmail.com
- LinkedIn: [David Babu](https://www.linkedin.com/in/david-babu-15047096/)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.