import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# --- Configuration ---
try:
    # Configure the Gemini API with the key from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    logger.info("Configuring Gemini API")
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully")
    
except Exception as e:
    logger.critical(f"Error configuring Gemini API: {e}")
    print(f"Error configuring Gemini API: {e}")


def get_model(temperature=0.7):
    """
    Creates and returns a Gemini model with specified temperature.
    
    Args:
        temperature (float): Controls randomness in output (0.0-1.0)
    
    Returns:
        GenerativeModel: Configured Gemini model
    """
    return genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={
            'temperature': temperature,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 8192,
        }
    )


# --- Core Functions ---

def summarize_text(text, length="standard", include_key_points=True, include_topics=True, temperature=0.7):
    """
    Uses the Gemini API to create a comprehensive, structured summary of the provided text.

    Args:
        text (str): The text to be summarized.
        length (str): The detail level - "brief", "standard", or "detailed".
        include_key_points (bool): Whether to include a key points section.
        include_topics (bool): Whether to include main topics section.
        temperature (float): AI creativity level.

    Returns:
        str: The comprehensive summarized text in markdown format, or an error message.
    """
    logger.info(f"Summarization requested - Length: {length}, Text length: {len(text)} chars")
    
    try:
        model = get_model(temperature)
        
        # Determine the summary length parameters
        length_instructions = {
            "brief": "Provide a concise summary in 3-5 sentences focusing only on the most critical information.",
            "standard": "Provide a well-balanced summary in 2-3 paragraphs that covers main points and supporting details.",
            "detailed": "Provide a comprehensive and thorough summary in 4-6 paragraphs that captures all significant information, nuances, and context."
        }
        
        # Build the prompt dynamically
        prompt = f"""You are an expert document analyst. Your task is to create a high-quality, structured summary of the following document.

{length_instructions.get(length, length_instructions["standard"])}

**Instructions:**
1. Read and analyze the entire document carefully
2. Identify the core themes, arguments, and conclusions
3. Maintain objectivity and accuracy
4. Use clear, professional language
5. Structure your response in markdown format

**Format your response as follows:**

## Executive Summary
[{length_instructions.get(length, length_instructions["standard"])}]

"""
        
        if include_topics:
            prompt += """
## Main Topics Covered
[List 3-5 main topics or themes discussed in the document as bullet points]

"""
        
        if include_key_points:
            prompt += """
## Key Points & Insights
[List 5-7 most important points, findings, or takeaways as bullet points]

"""
        
        prompt += """
## Conclusion
[Briefly summarize the overall significance or conclusion of the document in 2-3 sentences]

---

**Document to Summarize:**

"""
        prompt += f"\n{text}\n"
        
        logger.info(f"Sending summarization request to Gemini API, prompt length: {len(prompt)} chars")
        response = model.generate_content(prompt)
        logger.info(f"Summary generated successfully, response length: {len(response.text)} chars")
        return response.text
        
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        return f"An error occurred during summarization: {e}"


def answer_question(context, question, chat_history=None, temperature=0.7):
    """
    Uses the Gemini API to answer a question based on the provided context,
    maintaining conversation history for coherent multi-turn dialogues.

    Args:
        context (str): The relevant document sections from vector search.
        question (str): The user's current question.
        chat_history (list): List of previous messages in the conversation.
        temperature (float): AI creativity level.

    Returns:
        str: The answer to the question, or an error message.
    """
    logger.info(f"Q&A requested - Question: '{question[:100]}...', Context length: {len(context)} chars")
    
    try:
        model = get_model(temperature)
        
        # Build conversation context from history
        conversation_context = ""
        if chat_history and len(chat_history) > 1:
            # Include last few exchanges for context (excluding the current question)
            recent_history = chat_history[-6:-1] if len(chat_history) > 6 else chat_history[:-1]
            if recent_history:
                logger.info(f"Including {len(recent_history)} previous messages in context")
                conversation_context = "\n**Previous Conversation:**\n"
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_context += f"{role}: {msg['content']}\n"
                conversation_context += "\n"
        
        # Create a comprehensive prompt for Q&A
        prompt = f"""You are an intelligent document analysis assistant. Your role is to answer questions based strictly on the provided document content.

**Guidelines:**
1. Answer ONLY based on information explicitly stated or clearly implied in the document
2. If the answer cannot be found in the document, clearly state: "I cannot find this information in the provided document."
3. Provide specific, accurate, and helpful responses
4. Quote relevant sections when appropriate (use quotation marks)
5. If the question relates to previous conversation, maintain context continuity
6. Be concise but thorough - aim for clarity over brevity
7. If the question is ambiguous, address the most likely interpretation

{conversation_context}

**Relevant Document Sections:**
---
{context}
---

**Current Question:** {question}

**Your Answer:**"""
        
        logger.info(f"Sending Q&A request to Gemini API, prompt length: {len(prompt)} chars")
        response = model.generate_content(prompt)
        logger.info(f"Answer generated successfully, response length: {len(response.text)} chars")
        return response.text
        
    except Exception as e:
        logger.error(f"Error during Q&A: {e}", exc_info=True)
        return f"An error occurred while answering the question: {e}"


def extract_entities(text, temperature=0.7):
    """
    Extracts key entities from the document (people, organizations, locations, dates, etc.)
    
    Args:
        text (str): The document text.
        temperature (float): AI creativity level.
    
    Returns:
        str: Formatted entity extraction results.
    """
    logger.info("Entity extraction requested")
    
    try:
        model = get_model(temperature)
        
        prompt = f"""Analyze the following document and extract key entities. Organize them into categories:

**Instructions:**
- Extract named entities with high confidence
- Categorize entities appropriately
- Provide context for important entities
- Format response in markdown

**Categories to extract:**
1. **People**: Names of individuals mentioned
2. **Organizations**: Companies, institutions, groups
3. **Locations**: Places, cities, countries, regions
4. **Dates & Time Periods**: Specific dates, years, time ranges
5. **Products/Technologies**: Specific products, tools, or technologies
6. **Key Concepts**: Important ideas or terminology

**Document:**
---
{text[:4000]}
---

Provide a structured extraction with brief context for each entity."""

        response = model.generate_content(prompt)
        logger.info("Entity extraction completed")
        return response.text
        
    except Exception as e:
        logger.error(f"Error during entity extraction: {e}", exc_info=True)
        return f"An error occurred during entity extraction: {e}"


def analyze_sentiment(text, temperature=0.7):
    """
    Analyzes the sentiment and tone of the document.
    
    Args:
        text (str): The document text.
        temperature (float): AI creativity level.
    
    Returns:
        str: Sentiment analysis results.
    """
    logger.info("Sentiment analysis requested")
    
    try:
        model = get_model(temperature)
        
        prompt = f"""Perform a comprehensive sentiment analysis on the following document:

**Analysis Requirements:**
1. **Overall Sentiment**: Positive, Negative, Neutral, or Mixed
2. **Tone**: Formal, Informal, Academic, Persuasive, etc.
3. **Emotional Indicators**: Key phrases that reveal sentiment
4. **Objectivity Level**: How factual vs. opinion-based is the content
5. **Target Audience**: Who is this document written for
6. **Writing Style**: Descriptive, analytical, narrative, etc.

**Document:**
---
{text[:4000]}
---

Provide a detailed sentiment analysis in markdown format."""

        response = model.generate_content(prompt)
        logger.info("Sentiment analysis completed")
        return response.text
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
        return f"An error occurred during sentiment analysis: {e}"


def extract_key_terms(text, temperature=0.7):
    """
    Extracts and explains key terms and concepts from the document.
    
    Args:
        text (str): The document text.
        temperature (float): AI creativity level.
    
    Returns:
        str: Key terms with definitions.
    """
    logger.info("Key terms extraction requested")
    
    try:
        model = get_model(temperature)
        
        prompt = f"""Identify and explain the most important terms and concepts in this document:

**Instructions:**
1. Extract 10-15 key terms or concepts
2. Provide a brief definition or explanation for each
3. Indicate why each term is important to understanding the document
4. Organize by importance or theme

**Document:**
---
{text[:4000]}
---

Format as a markdown list with term name in bold, followed by explanation."""

        response = model.generate_content(prompt)
        logger.info("Key terms extraction completed")
        return response.text
        
    except Exception as e:
        logger.error(f"Error during key terms extraction: {e}", exc_info=True)
        return f"An error occurred during key terms extraction: {e}"


def compare_documents(text1, text2, temperature=0.7):
    """
    Compares two documents and identifies similarities and differences.
    
    Args:
        text1 (str): First document text.
        text2 (str): Second document text.
        temperature (float): AI creativity level.
    
    Returns:
        str: Comparison analysis.
    """
    logger.info("Document comparison requested")
    
    try:
        model = get_model(temperature)
        
        prompt = f"""Compare and contrast these two documents:

**Comparison Framework:**
1. **Common Themes**: What topics/ideas do both documents share?
2. **Key Differences**: How do they differ in content, perspective, or approach?
3. **Tone & Style**: Compare writing style and formality
4. **Unique Content**: What is unique to each document?
5. **Contradictions**: Are there any contradictory claims?
6. **Complementarity**: How might these documents complement each other?

**Document 1:**
---
{text1[:3000]}
---

**Document 2:**
---
{text2[:3000]}
---

Provide a structured comparison in markdown format."""

        response = model.generate_content(prompt)
        logger.info("Document comparison completed")
        return response.text
        
    except Exception as e:
        logger.error(f"Error during document comparison: {e}", exc_info=True)
        return f"An error occurred during document comparison: {e}"


def generate_full_report(text, document_name, temperature=0.7):
    """
    Generates a comprehensive analysis report of the document.
    
    Args:
        text (str): The document text.
        document_name (str): Name of the document.
        temperature (float): AI creativity level.
    
    Returns:
        str: Full comprehensive report.
    """
    logger.info("Full report generation requested")
    
    try:
        model = get_model(temperature)
        
        prompt = f"""Generate a comprehensive analysis report for the following document:

**Report Structure:**

# Document Analysis Report
**Document:** {document_name}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Executive Summary
[Provide a high-level overview of the document in 3-4 paragraphs]

## 2. Document Metadata
- **Length**: [word count and page estimate]
- **Type**: [document type/genre]
- **Complexity**: [reading level]

## 3. Main Topics
[List and explain 5-7 main topics covered]

## 4. Key Findings
[List 8-10 most important findings or points]

## 5. Supporting Evidence
[Describe types of evidence or support provided]

## 6. Conclusions
[What conclusions can be drawn from this document?]

## 7. Recommendations
[If applicable, what actions or next steps are suggested?]

## 8. Questions for Further Investigation
[List 5 questions that readers might want to explore further]

---

**Document Content:**
{text[:5000]}

Provide a thorough, professional analysis report."""

        response = model.generate_content(prompt)
        logger.info("Full report generation completed")
        return response.text
        
    except Exception as e:
        logger.error(f"Error during report generation: {e}", exc_info=True)
        return f"An error occurred during report generation: {e}"