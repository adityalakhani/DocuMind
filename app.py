import streamlit as st
import os
import logging
from datetime import datetime
from modules import ai_core, document_processor

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📚"
)

logger.info("Application started")

# --- Custom CSS for better UI ---
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .message-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stTextInput input {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    **AI Document Analyzer** leverages Google's Gemini AI with RAG (Retrieval Augmented Generation) to provide:
    
    - **Comprehensive Summaries**: Get detailed, structured summaries with key insights
    - **Interactive Q&A**: Chat naturally with your documents using semantic search
    - **Context-Aware Responses**: Answers grounded in relevant document sections
    - **Smart Chunking**: Efficient processing of large documents
    - **Export Capabilities**: Save summaries and chat history
    
    ### How to use:
    1. Upload a document (PDF or TXT)
    2. Choose your desired action
    3. Interact with AI-powered insights
    """)
    
    st.divider()
    
    # Document info and settings
    if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
        st.subheader("Document Information")
        
        word_count = len(st.session_state.document_text.split())
        char_count = len(st.session_state.document_text)
        chunk_count = st.session_state.vector_store._collection.count()
        
        st.metric("Document Words", f"{word_count:,}")
        st.metric("Document Characters", f"{char_count:,}")
        st.metric("Vector Chunks", f"{chunk_count:,}")
        
        st.divider()
        st.subheader("Settings")
        
        # Retrieval settings
        st.session_state.retrieval_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.get('retrieval_k', 3),
            help="More chunks = more context but slower processing"
        )
        
        # Temperature setting
        st.session_state.ai_temperature = st.slider(
            "AI Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('ai_temperature', 0.7),
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        st.divider()
        
        if st.button("Clear Document", use_container_width=True):
            logger.info("Clearing document and chat history")
            st.session_state.document_text = ""
            st.session_state.chat_history = []
            st.session_state.vector_store = None
            st.session_state.uploaded_filename = None
            st.rerun()

# --- Main App Logic ---
st.title("AI Document Analyzer")
st.markdown("*Powered by Google Gemini AI with RAG*")

# Initialize session state
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retrieval_k' not in st.session_state:
    st.session_state.retrieval_k = 3
if 'ai_temperature' not in st.session_state:
    st.session_state.ai_temperature = 0.7
if 'summary_cache' not in st.session_state:
    st.session_state.summary_cache = {}

# Check for API Key
if not os.getenv("GEMINI_API_KEY"):
    logger.error("GEMINI_API_KEY not found in environment variables")
    st.error("GEMINI_API_KEY is not set. Please add it to your .env file.")
    st.info("Create a `.env` file in your project root with: `GEMINI_API_KEY=your_api_key_here`")
else:
    logger.info("API key found, application ready")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file to get started",
        type=["pdf", "txt"],
        help="The content will be analyzed by AI using semantic chunking and vector search."
    )

    if uploaded_file:
        # Check if a new file was uploaded
        if st.session_state.uploaded_filename != uploaded_file.name:
            logger.info(f"New file uploaded: {uploaded_file.name} (type: {uploaded_file.type})")
            file_type = uploaded_file.type
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                if file_type == "application/pdf":
                    logger.info(f"Processing PDF file: {uploaded_file.name}")
                    st.session_state.document_text = document_processor.get_text_from_pdf(uploaded_file)
                elif file_type == "text/plain":
                    logger.info(f"Processing TXT file: {uploaded_file.name}")
                    st.session_state.document_text = document_processor.get_text_from_txt(uploaded_file)
                
                if "Error" in st.session_state.document_text:
                    logger.error(f"Error processing file: {st.session_state.document_text}")
                    st.error(st.session_state.document_text)
                    st.session_state.document_text = ""
                else:
                    # Create vector store with semantic chunking
                    with st.spinner("Creating vector embeddings..."):
                        st.session_state.vector_store = document_processor.create_vector_store(
                            st.session_state.document_text,
                            uploaded_file.name
                        )
                    
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.chat_history = []
                    st.session_state.summary_cache = {}
                    logger.info("Chat history cleared for new document")
                    logger.info(f"File processed successfully: {uploaded_file.name}, {len(st.session_state.document_text)} characters")
                    st.success(f"{uploaded_file.name} processed successfully!")
                    st.rerun()

    # Display processing options only if document is loaded
    if st.session_state.vector_store is not None:
        st.divider()
        
        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "Comprehensive Summary", 
            "Interactive Q&A Chat", 
            "Document Analysis",
            "Export & History"
        ])
        
        # TAB 1: SUMMARY
        with tab1:
            st.subheader("Document Summary")
            st.markdown("Generate an in-depth, structured summary of your document.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                summary_length = st.selectbox(
                    "Summary Detail Level",
                    ["Brief", "Standard", "Detailed"],
                    index=1
                )
            with col2:
                include_key_points = st.checkbox("Include Key Points", value=True)
            with col3:
                include_topics = st.checkbox("Include Main Topics", value=True)
            
            # Cache key for summary
            cache_key = f"{summary_length}_{include_key_points}_{include_topics}"
            
            if st.button("Generate Summary", use_container_width=True, type="primary"):
                # Check cache first
                if cache_key in st.session_state.summary_cache:
                    logger.info("Using cached summary")
                    summary = st.session_state.summary_cache[cache_key]
                    st.info("Using cached summary (regenerate by changing options)")
                else:
                    logger.info(f"Summary requested - Length: {summary_length}, Key Points: {include_key_points}, Topics: {include_topics}")
                    with st.spinner("AI is analyzing your document..."):
                        summary = ai_core.summarize_text(
                            st.session_state.document_text,
                            length=summary_length.lower(),
                            include_key_points=include_key_points,
                            include_topics=include_topics,
                            temperature=st.session_state.ai_temperature
                        )
                        st.session_state.summary_cache[cache_key] = summary
                        logger.info(f"Summary generated successfully, {len(summary)} characters")
                
                st.success("Summary Generated!")
                st.markdown(summary)
                
                # Download option
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"summary_{st.session_state.uploaded_filename}.txt",
                    mime="text/plain"
                )
        
        # TAB 2: Q&A CHAT
        with tab2:
            st.subheader("Chat with Your Document")
            st.markdown("Ask questions and get context-aware answers from your document.")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <div class="message-label">You:</div>
                            <div>{message["content"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <div class="message-label">AI Assistant:</div>
                            <div>{message["content"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show sources if available
                        if "sources" in message and message["sources"]:
                            with st.expander("View relevant document sections"):
                                for idx, source in enumerate(message["sources"], 1):
                                    st.markdown(f"**Section {idx}:**")
                                    st.text(source[:500] + "..." if len(source) > 500 else source)
            
            # Chat input form
            with st.form(key="chat_form", clear_on_submit=True):
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    user_question = st.text_input(
                        "Ask your question:",
                        placeholder="e.g., What are the main conclusions? Can you explain...?",
                        key="question_input",
                        label_visibility="collapsed"
                    )
                
                with col2:
                    send_button = st.form_submit_button("Send", use_container_width=True, type="primary")
            
            if send_button and user_question:
                logger.info(f"User question: {user_question[:100]}...")
                
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                
                # Get relevant context from vector store
                with st.spinner("Searching document..."):
                    relevant_chunks = st.session_state.vector_store.similarity_search(
                        user_question, 
                        k=st.session_state.retrieval_k
                    )
                    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
                    logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
                
                # Get AI response
                with st.spinner("AI is thinking..."):
                    answer = ai_core.answer_question(
                        context,
                        user_question,
                        st.session_state.chat_history,
                        temperature=st.session_state.ai_temperature
                    )
                    logger.info(f"Answer generated, {len(answer)} characters")
                    
                    # Add assistant response to history with sources
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [chunk.page_content for chunk in relevant_chunks]
                    })
                
                st.rerun()
            
            # Clear chat history button
            if st.session_state.chat_history:
                if st.button("Clear Chat History"):
                    logger.info("Chat history cleared by user")
                    st.session_state.chat_history = []
                    st.rerun()
        
        # TAB 3: DOCUMENT ANALYSIS
        with tab3:
            st.subheader("Document Analysis Tools")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Entity Extraction")
                st.markdown("Extract key entities from your document")
                if st.button("Extract Entities", use_container_width=True):
                    with st.spinner("Analyzing document..."):
                        entities = ai_core.extract_entities(
                            st.session_state.document_text,
                            temperature=st.session_state.ai_temperature
                        )
                        st.markdown(entities)
            
            with col2:
                st.markdown("#### Sentiment Analysis")
                st.markdown("Analyze the overall tone and sentiment")
                if st.button("Analyze Sentiment", use_container_width=True):
                    with st.spinner("Analyzing sentiment..."):
                        sentiment = ai_core.analyze_sentiment(
                            st.session_state.document_text,
                            temperature=st.session_state.ai_temperature
                        )
                        st.markdown(sentiment)
            
            st.divider()
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### Key Terms")
                st.markdown("Identify important terminology and concepts")
                if st.button("Extract Key Terms", use_container_width=True):
                    with st.spinner("Extracting terms..."):
                        terms = ai_core.extract_key_terms(
                            st.session_state.document_text,
                            temperature=st.session_state.ai_temperature
                        )
                        st.markdown(terms)
            
            with col4:
                st.markdown("#### Document Comparison")
                st.markdown("Compare with another document")
                compare_file = st.file_uploader(
                    "Upload second document",
                    type=["pdf", "txt"],
                    key="compare_upload"
                )
                if compare_file and st.button("Compare Documents", use_container_width=True):
                    with st.spinner("Processing comparison..."):
                        if compare_file.type == "application/pdf":
                            compare_text = document_processor.get_text_from_pdf(compare_file)
                        else:
                            compare_text = document_processor.get_text_from_txt(compare_file)
                        
                        comparison = ai_core.compare_documents(
                            st.session_state.document_text,
                            compare_text,
                            temperature=st.session_state.ai_temperature
                        )
                        st.markdown(comparison)
        
        # TAB 4: EXPORT & HISTORY
        with tab4:
            st.subheader("Export & History")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Export Chat History")
                if st.session_state.chat_history:
                    # Format chat history for export
                    chat_export = ""
                    for msg in st.session_state.chat_history:
                        role = "You" if msg["role"] == "user" else "AI Assistant"
                        chat_export += f"{role}: {msg['content']}\n\n"
                    
                    st.download_button(
                        label="Download Chat History",
                        data=chat_export,
                        file_name=f"chat_history_{st.session_state.uploaded_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.info("No chat history available")
            
            with col2:
                st.markdown("#### Export Full Report")
                if st.button("Generate Full Report", use_container_width=True):
                    with st.spinner("Generating comprehensive report..."):
                        report = ai_core.generate_full_report(
                            st.session_state.document_text,
                            st.session_state.uploaded_filename,
                            temperature=st.session_state.ai_temperature
                        )
                        
                        st.download_button(
                            label="Download Full Report",
                            data=report,
                            file_name=f"full_report_{st.session_state.uploaded_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            
            st.divider()
            
            st.markdown("#### Session Statistics")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Total Questions Asked", len([m for m in st.session_state.chat_history if m["role"] == "user"]))
            
            with col4:
                st.metric("Summaries Generated", len(st.session_state.summary_cache))
            
            with col5:
                st.metric("Document Chunks", st.session_state.vector_store._collection.count() if st.session_state.vector_store else 0)
    else:
        st.info("Please upload a document to begin analysis.")