import PyPDF2
import logging
from io import BytesIO
from typing import List
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.

    Args:
        pdf_file (streamlit.UploadedFile): The uploaded PDF file object.

    Returns:
        str: The extracted text, or an error message.
    """
    logger.info(f"Starting PDF text extraction for file: {pdf_file.name}")
    
    try:
        # PyPDF2 requires a file-like object that supports seek, so we use BytesIO
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.getvalue()))
        total_pages = len(pdf_reader.pages)
        logger.info(f"PDF has {total_pages} pages")
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
            else:
                logger.warning(f"No text found on page {page_num}")
        
        result = text.strip() if text else "Could not extract any text from the PDF."
        
        if text:
            logger.info(f"Successfully extracted {len(result)} characters from PDF")
        else:
            logger.warning("No text could be extracted from PDF")
            
        return result
        
    except Exception as e:
        error_msg = f"Error reading PDF file: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def get_text_from_txt(txt_file):
    """
    Extracts text from an uploaded TXT file.

    Args:
        txt_file (streamlit.UploadedFile): The uploaded TXT file object.

    Returns:
        str: The extracted text.
    """
    logger.info(f"Starting TXT file extraction for file: {txt_file.name}")
    
    try:
        # The file uploader reads the file as bytes, so we need to decode it
        text = txt_file.getvalue().decode("utf-8")
        logger.info(f"Successfully extracted {len(text)} characters from TXT file")
        return text
        
    except UnicodeDecodeError as e:
        # Try alternative encodings
        logger.warning(f"UTF-8 decoding failed, trying alternative encodings: {e}")
        try:
            text = txt_file.getvalue().decode("latin-1")
            logger.info(f"Successfully decoded with latin-1 encoding, {len(text)} characters")
            return text
        except Exception as e2:
            error_msg = f"Error reading TXT file with multiple encodings: {e2}"
            logger.error(error_msg, exc_info=True)
            return error_msg
            
    except Exception as e:
        error_msg = f"Error reading TXT file: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def semantic_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits text into semantic chunks using recursive character splitting.
    This method preserves semantic meaning by splitting on natural boundaries.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[Document]: List of LangChain Document objects containing text chunks.
    """
    logger.info(f"Starting semantic chunking - text length: {len(text)} chars")
    
    try:
        # Clean the text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r' {2,}', ' ', text)  # Remove excessive spaces
        
        # Create text splitter with semantic-aware separators
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple newlines (section breaks)
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence endings
                "! ",      # Exclamation endings
                "? ",      # Question endings
                "; ",      # Semicolon
                ", ",      # Comma
                " ",       # Space
                ""         # Character-level (fallback)
            ],
            is_separator_regex=False,
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} semantic chunks")
        return documents
        
    except Exception as e:
        logger.error(f"Error during semantic chunking: {e}", exc_info=True)
        # Fallback: return entire text as single chunk
        return [Document(page_content=text, metadata={"chunk_id": 0, "total_chunks": 1})]


def create_vector_store(text: str, document_name: str) -> Chroma:
    """
    Creates a ChromaDB vector store from text using semantic chunking and embeddings.

    Args:
        text (str): The document text to process.
        document_name (str): Name of the document for metadata.

    Returns:
        Chroma: ChromaDB vector store containing embedded document chunks.
    """
    logger.info(f"Creating vector store for document: {document_name}")
    
    try:
        # Step 1: Semantic chunking
        chunks = semantic_chunk_text(text)
        
        # Add document name to metadata
        for chunk in chunks:
            chunk.metadata["document_name"] = document_name
        
        logger.info(f"Created {len(chunks)} chunks for vector store")
        
        # Step 2: Create embeddings
        # Using a lightweight, efficient embedding model
        logger.info("Initializing embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Step 3: Create vector store
        logger.info("Creating ChromaDB vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"doc_{hash(document_name)}",
            collection_metadata={"document": document_name}
        )
        
        logger.info(f"Vector store created successfully with {len(chunks)} chunks")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise


def query_vector_store(vector_store: Chroma, query: str, k: int = 3) -> List[Document]:
    """
    Queries the vector store for relevant document chunks.

    Args:
        vector_store (Chroma): The ChromaDB vector store.
        query (str): The search query.
        k (int): Number of top results to return.

    Returns:
        List[Document]: List of most relevant document chunks.
    """
    logger.info(f"Querying vector store - Query: '{query[:100]}...', k={k}")
    
    try:
        results = vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} relevant chunks")
        return results
        
    except Exception as e:
        logger.error(f"Error querying vector store: {e}", exc_info=True)
        return []


def get_chunk_statistics(vector_store: Chroma) -> dict:
    """
    Gets statistics about the chunks in the vector store.

    Args:
        vector_store (Chroma): The ChromaDB vector store.

    Returns:
        dict: Dictionary containing chunk statistics.
    """
    try:
        collection = vector_store._collection
        count = collection.count()
        
        # Get sample of chunks to calculate average size
        sample = collection.get(limit=min(100, count))
        if sample and 'documents' in sample:
            avg_size = sum(len(doc) for doc in sample['documents']) / len(sample['documents'])
        else:
            avg_size = 0
        
        stats = {
            "total_chunks": count,
            "average_chunk_size": int(avg_size),
            "collection_name": collection.name
        }
        
        logger.info(f"Chunk statistics: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting chunk statistics: {e}", exc_info=True)
        return {"total_chunks": 0, "average_chunk_size": 0}