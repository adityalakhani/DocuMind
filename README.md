# DocuMind

**Powered by Google Gemini with Retrieval-Augmented Generation (RAG)**

An interactive Streamlit application that allows you to "chat" with your documents. Upload a PDF or TXT file and leverage the power of Google's Gemini large language model to get comprehensive summaries, ask questions, and perform in-depth analysis.

---

## Core Features

* **Comprehensive Summaries**: Get detailed, structured summaries with key insights at brief, standard, or detailed levels.
* **Interactive Q&A**: Chat naturally with your documents. The AI provides answers grounded in the document's content, citing the relevant sections used.
* **Advanced Document Analysis**:
    * **Entity Extraction**: Automatically identify and categorize key entities like people, organizations, and locations.
    * **Sentiment Analysis**: Analyze the overall tone and sentiment of the text.
    * **Key Term Extraction**: Identify and get explanations for important terminology and concepts.
    * **Document Comparison**: Compare the primary document with a second one to find similarities and differences.
* **Context-Aware Responses**: Utilizes Retrieval-Augmented Generation (RAG) to ensure answers are based on the document's content, reducing hallucinations.
* **Smart Chunking**: Employs semantic chunking to efficiently process large documents while preserving context.
* **Export Capabilities**: Easily download generated summaries, chat history, or a full analysis report.
* **Customizable AI Settings**: Adjust the AI's creativity (temperature) and the number of document chunks to retrieve for a tailored experience.

## How It Works

This application integrates several modern AI and data processing technologies to provide its features:

1.  **Frontend**: The user interface is built with **Streamlit**, providing a fast and interactive web app experience.
2.  **Document Processing**: Uploaded PDF and TXT files are parsed to extract raw text using libraries like `PyPDF2`.
3.  **Semantic Chunking**: The extracted text is split into smaller, overlapping chunks using `LangChain`'s `RecursiveCharacterTextSplitter`. This method attempts to split text along natural semantic boundaries (paragraphs, sentences) to maintain context within each chunk.
4.  **Vector Embeddings**: Each text chunk is converted into a numerical vector representation (embedding) using the `sentence-transformers/all-MiniLM-L6-v2` model. These embeddings capture the semantic meaning of the text.
5.  **Vector Store**: The embeddings are stored in a **ChromaDB** in-memory vector store. This allows for extremely fast and efficient semantic searching.
6.  **Retrieval-Augmented Generation (RAG)**:
    * **Retrieval**: When you ask a question, the application converts your query into an embedding and uses it to search the ChromaDB store. It retrieves the most semantically similar text chunks from the document.
    * **Augmentation**: These retrieved chunks are then combined with your original question and the chat history to form a detailed prompt.
    * **Generation**: This augmented prompt is sent to the **Google Gemini** API, which generates a contextually aware and accurate answer based *only* on the provided information.

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.8+
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/adityalakhani/DocuMind
cd DocuMind
