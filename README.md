Financial Chatbot for Infosys Financial Reports
------------------------------------------------
- This is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Infosys financial statements from the last two years (2022-2024). 
- The chatbot uses open-source models and advanced retrieval techniques to provide accurate and concise answers.

Project Structure
------------------
- The project is organized as follows:
```
Financial-Chatbot/
├── app.py                  # Streamlit application interface
├── chroma_db/              # Chroma vector database storage
├── Infy financial report/  # Folder containing Infosys financial PDFs
│   ├── INFY_2022_2023.pdf
│   └── INFY_2023_2024.pdf
├── requirements.txt        # Python dependencies
├── utils.py                # Core functionality and RAG implementation
└── README.md               # This file
```

Installation
--------------
Python Version: ```Python 3.10.xx```

Python lib requirements: ```pip install -r requirements.txt```


Place PDFs:
------------
- Ensure the Infosys financial reports (INFY_2022_2023.pdf and INFY_2023_2024.pdf) are placed in the Infy financial report/ folder.


Running the Application
------------------------
- To start the chatbot, run the following command:

```streamlit run app.py --server.enableCORS false```

- The application will start and provide a local URL (e.g., http://localhost:8501). Open this URL in your browser to interact with the chatbot.


