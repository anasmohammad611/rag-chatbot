import logging
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Import our vector database
from vectorDB import ChromaVectorDB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AngelOne RAG Chatbot (Free)",
    description="AI chatbot for AngelOne support queries using free models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and database
vector_db = None
qa_pipeline = None
text_generator = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    vector_db_status: str
    total_chunks: int
    model_status: str


def load_free_models():
    """Load free Hugging Face models"""
    global qa_pipeline, text_generator

    try:
        logger.info("Loading free AI models...")

        # Option 1: Question Answering pipeline (lightweight)
        logger.info("Loading Q&A model...")
        # qa_pipeline = pipeline(
        #     "question-answering",
        #     model="distilbert-base-cased-distilled-squad",
        #     tokenizer="distilbert-base-cased-distilled-squad"
        # )
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad",  # Smaller variant
            tokenizer="distilbert-base-uncased-distilled-squad"
        )

        # Option 2: Text generation model (for better responses)
        logger.info("Loading text generation model...")
        # try:
        #     # Try a small, fast model first
        #     text_generator = pipeline(
        #         "text-generation",
        #         model="microsoft/DialoGPT-small",
        #         tokenizer="microsoft/DialoGPT-small",
        #         max_length=200,
        #         do_sample=True,
        #         temperature=0.7,
        #         pad_token_id=50256
        #     )
        # except Exception as e:
        #     logger.warning(f"Could not load text generation model: {e}")
        #     text_generator = None

        text_generator = None  # Disable text generation for prod

        logger.info("✅ Models loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


def generate_response_with_context(query: str, context_chunks: List[Dict]) -> str:
    """Generate response using free models with retrieved context"""

    if not context_chunks:
        return "I Don't know"

    try:
        # Prepare context
        context_text = ""
        for chunk in context_chunks[:3]:  # Use top 3 chunks
            context_text += chunk['text'] + " "

        # Limit context length to avoid model limits
        context_text = context_text[:1500]

        # Method 1: Use Question Answering pipeline
        if qa_pipeline:
            try:
                result = qa_pipeline(
                    question=query,
                    context=context_text
                )

                # Check confidence score
                if result['score'] > 0.1:  # Minimum confidence threshold
                    answer = result['answer']

                    # Add source context to make answer more complete
                    if len(answer.split()) < 10:  # If answer is too short
                        # Try to expand with relevant context
                        sentences = context_text.split('.')
                        for sentence in sentences:
                            if any(word.lower() in sentence.lower() for word in query.split()[:3]):
                                expanded = sentence.strip()
                                if len(expanded) > len(answer):
                                    answer = expanded
                                break

                    return answer if answer else "I Don't know"

            except Exception as e:
                logger.warning(f"Q&A pipeline failed: {e}")

        # Method 2: Simple keyword-based response (fallback)
        return generate_simple_response(query, context_chunks)

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I Don't know"


def generate_simple_response(query: str, context_chunks: List[Dict]) -> str:
    """Simple keyword-based response generation (fallback)"""

    query_lower = query.lower()

    # Find the most relevant chunk
    best_chunk = context_chunks[0] if context_chunks else None
    if not best_chunk:
        return "I Don't know"

    text = best_chunk['text']

    # Extract relevant sentences based on query keywords
    sentences = text.split('.')
    relevant_sentences = []

    query_words = [word.strip('?.,!') for word in query_lower.split()]

    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if sentence contains query keywords
        word_matches = sum(1 for word in query_words if word in sentence_lower)
        if word_matches >= 2 or len(query_words) <= 2 and word_matches >= 1:
            relevant_sentences.append(sentence.strip())

    if relevant_sentences:
        # Return the most relevant sentence(s)
        response = '. '.join(relevant_sentences[:2])  # Max 2 sentences
        return response if len(response) > 20 else text[:200] + "..."

    # If no specific match, return beginning of most relevant chunk
    return text[:200] + "..."


# Initialize everything on startup
@app.on_event("startup")
async def startup_event():
    global vector_db

    try:
        logger.info("🚀 Starting AngelOne RAG Chatbot...")

        # Initialize vector database
        logger.info("Initializing vector database...")
        vector_db = ChromaVectorDB()
        stats = vector_db.get_stats()

        if stats['total_chunks'] == 0:
            logger.warning("⚠️ No chunks found in vector database. Please run vector_db.py first.")
        else:
            logger.info(f"✅ Vector database ready with {stats['total_chunks']} chunks")

        # Load AI models
        model_loaded = load_free_models()
        if not model_loaded:
            logger.warning("⚠️ AI models not loaded. Responses will be basic.")

        logger.info("🎉 System ready!")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AngelOne RAG Chatbot API (Free Version)",
        "status": "active",
        "models": "Hugging Face Transformers",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if vector_db is None:
        return HealthResponse(
            status="unhealthy",
            vector_db_status="not_initialized",
            total_chunks=0,
            model_status="not_loaded"
        )

    stats = vector_db.get_stats()
    model_status = "loaded" if qa_pipeline else "basic"

    return HealthResponse(
        status="healthy",
        vector_db_status=stats["status"],
        total_chunks=stats["total_chunks"],
        model_status=model_status
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for RAG queries"""

    # Validate request
    if not request.message or len(request.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Check if vector database is available
    if vector_db is None:
        raise HTTPException(
            status_code=503,
            detail="Vector database not available. Please run vector_db.py first."
        )

    try:
        # Step 1: Search for relevant chunks
        logger.info(f"Processing query: {request.message}")
        relevant_chunks = vector_db.search(
            query=request.message,
            top_k=5,
            min_similarity=0.42  # Optimal threshold based on testing
        )

        # Step 2: Check if we found relevant context
        if not relevant_chunks:
            logger.info("No relevant context found")
            return ChatResponse(
                response="I Don't know",
                sources=[],
                conversation_id=request.conversation_id
            )

        # Step 3: Generate response using free models
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")
        response_text = generate_response_with_context(request.message, relevant_chunks)

        # Step 4: Prepare sources for response
        sources = []
        seen_urls = set()
        for chunk in relevant_chunks:
            url = chunk['metadata']['source_url']
            if url not in seen_urls:
                source = {
                    "title": chunk['metadata']['title'],
                    "url": url,
                    "relevance_score": round(chunk['similarity_score'], 3)
                }
                sources.append(source)
                seen_urls.add(url)

        logger.info(f"Generated response with {len(sources)} sources")

        return ChatResponse(
            response=response_text,
            sources=sources[:3],  # Limit to top 3 sources
            conversation_id=request.conversation_id
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return ChatResponse(
            response="I Don't know",
            sources=[],
            conversation_id=request.conversation_id
        )


@app.get("/api/test")
async def test_rag():
    """Test the RAG system with sample queries"""
    if vector_db is None:
        return {"error": "Vector database not initialized"}

    test_queries = [
        "How do I add funds to my trading account?",
        "What are the brokerage fees for equity trading?",
        "How to cook biryani recipe?",
        "How to invest in stock market?",
        "What is margin trading facility in AngelOne?",
        "What is the GDP of India in 2024?",
        "How do I withdraw money from my account?",
        "Who is Elon Musk?"
    ]

    results = {}
    for query in test_queries:
        try:
            chunks = vector_db.search(query, top_k=3, min_similarity=0.42)
            if chunks:
                response = generate_response_with_context(query, chunks)
                results[query] = {
                    "response": response,
                    "sources_found": len(chunks),
                    "top_source": chunks[0]['metadata']['title'] if chunks else None
                }
            else:
                results[query] = {
                    "response": "I don't know",
                    "sources_found": 0,
                    "top_source": None
                }
        except Exception as e:
            results[query] = {"error": str(e)}

    return {"test_results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
