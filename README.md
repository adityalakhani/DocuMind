# AI Document Analyzer

**Powered by Google Gemini + Retrieval-Augmented Generation (RAG)**

A Streamlit-based application that allows users to interact with documents conversationally. Upload PDF or TXT files to receive intelligent summaries, ask questions, extract entities, perform sentiment analysis, and more — all grounded in the document’s content.

---

## Features

- **Comprehensive Summaries**  
  Generate concise or detailed summaries highlighting key insights.

- **Interactive Q&A**  
  Ask questions and receive answers strictly based on document content, with citations.

- **Advanced Document Analysis**  
  - **Entity Extraction** — Identify people, organizations, locations, etc.  
  - **Sentiment Analysis** — Understand the emotional tone of the text.  
  - **Key Term Extraction** — Highlight important terms and get explanations.  
  - **Document Comparison** — Compare two documents for similarities and differences.

- **Context-Aware Responses**  
  Uses Retrieval-Augmented Generation (RAG) to ensure accurate, grounded answers.

- **Smart Chunking**  
  Semantic chunking of documents ensures better context preservation.

- **Export Results**  
  Export summaries, chat transcripts, or full analysis reports.

- **Customizable AI Settings**  
  Adjust creativity level (temperature) and retrieval parameters.

---

## How It Works

1. **Frontend**: Built using [Streamlit](https://streamlit.io/) for a responsive, interactive interface.
2. **Document Parsing**: Uses `PyPDF2` to extract text from PDFs or reads TXT files directly.
3. **Semantic Chunking**:  
   Utilizes `LangChain`'s `RecursiveCharacterTextSplitter` for semantically coherent text chunks.
4. **Embeddings**:  
   Converts text chunks to vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
5. **Vector Store**:  
   Embeddings are stored in `ChromaDB` for fast semantic retrieval.
6. **RAG Pipeline**:  
   - **Retrieval**: Converts your question into an embedding to fetch relevant chunks.  
   - **Augmentation**: Constructs a prompt using the retrieved chunks and your query.  
   - **Generation**: Sends the prompt to Google Gemini to generate an accurate response.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Google Gemini API Key

### 1. Clone the Repository

```bash
git clone https://github.com/adityalakhani/DocuMind
cd DocuMind
```

### 2. Create a Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add your API key:

```env
GEMINI_API_KEY="your_google_api_key_here"
```

Use `.env.example` as a reference.

### 5. Run the Application

```bash
streamlit run app.py
```

Access the app at: [http://localhost:8501](http://localhost:8501)

---

## Usage Guide

1. **Upload a Document**  
   Select a `.pdf` or `.txt` file using the file uploader.

2. **Processing**  
   The app extracts text, chunks it semantically, and creates embeddings.

3. **Explore Tabs**  
   - **Summary**: Choose detail level, click "Generate Summary".  
   - **Chat**: Ask questions; receive grounded, cited answers.  
   - **Analysis**: Run entity extraction, sentiment, or key term analysis.  
   - **Compare**: Upload a second document for side-by-side comparison.  
   - **Export**: Download summary, chat history, or analysis report.  
   - **Reset**: Clear the current document and start fresh.

---

## Project Structure

```
.
├── app.py                   # Main Streamlit app
├── requirements.txt         # Python dependencies
├── .env.example             # Example of environment variable setup
├── modules/
│   ├── __init__.py
│   ├── ai_core.py           # Handles Gemini API integration
│   └── document_processor.py # File parsing, chunking, embedding logic
```

---

## Technologies Used

| Category              | Tools / Libraries                             |
|----------------------|-----------------------------------------------|
| Frontend             | Streamlit                                     |
| AI & ML              | Google Generative AI, Sentence-Transformers   |
| Document Parsing     | PyPDF2                                        |
| Vector Storage       | ChromaDB                                      |
| Chunking             | LangChain                                     |
| Env Management       | python-dotenv                                 |

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## FAQ

**Q: What file types are supported?**  
A: Currently supports `.pdf` and `.txt`.

**Q: Does it work offline?**  
A: No, it requires an active internet connection and a valid Google Gemini API key.

**Q: Is it open-source?**  
A: Yes. Contributions are welcome.
