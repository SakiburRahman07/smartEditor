from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from chat_service import ChatService
from embedding_service import EmbeddingService
import google.generativeai as genai
import logging
import asyncio

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

chat_service = ChatService()
embedding_service = EmbeddingService()

logger = logging.getLogger(__name__)

async def cleanup_task():
    """Periodic task to clean up expired caches"""
    while True:
        await embedding_service.cleanup_expired_caches()
        await asyncio.sleep(3600)  # Run every hour

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(cleanup_task())

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    response = await chat_service.get_response(message)
    if "error" in response:
        raise HTTPException(status_code=500, detail=response["error"])
    return response

@app.get("/conversation/{cache_name}")
async def get_conversation(cache_name: str):
    try:
        cache = await embedding_service.get_cached_conversation(cache_name)
        return {"conversation": cache.contents[0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/test-cache")
async def test_cache():
    try:
        # Test conversation
        test_message = "Hello, how are you?"
        
        # Get initial response and cache
        response1 = await chat_service.get_response(test_message)
        if "error" in response1:
            return {"status": "error", "message": "Failed to get initial response"}
            
        cache_name = response1.get("cache_name")
        if not cache_name:
            return {"status": "error", "message": "No cache name received"}
            
        # Try to retrieve the cached conversation
        cached_conv = await embedding_service.get_cached_conversation(cache_name)
        if not cached_conv:
            return {"status": "error", "message": "Failed to retrieve cache"}
            
        # Verify cache contents
        cache_test_results = {
            "status": "success",
            "original_response": response1["response"],
            "cached_conversation": cached_conv.contents[0]["text"],
            "cache_name": cache_name,
            "cache_expiry": cached_conv.expire_time,
        }
        
        return cache_test_results
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cache test failed: {str(e)}"
        }

@app.get("/list-caches")
async def list_caches():
    """List all cached conversations"""
    try:
        caches = await embedding_service.list_caches()
        return {
            "status": "success",
            "total_caches": len(caches),
            "caches": caches
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list caches: {str(e)}"
        }

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse("test_cache.html", {"request": request})

@app.get("/similar-conversations/{message}")
async def find_similar(message: str):
    """Find similar conversations for a message"""
    try:
        embedding = await embedding_service.create_embedding(message)
        if not embedding:
            return {"status": "error", "message": "Failed to create embedding"}
            
        similar = await embedding_service.find_similar_conversations(embedding)
        
        return {
            "status": "success",
            "similar_conversations": [
                {
                    "cache_id": cache_id,
                    "similarity": similarity,
                    "conversation": await embedding_service.get_cached_conversation(cache_id)
                }
                for cache_id, similarity in similar
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-caching")
async def test_caching():
    """Test the complete caching system"""
    try:
        # Test 1: Create and cache a conversation
        test_message = "What is the weather today?"
        response1 = await chat_service.get_response(test_message)
        
        if not response1.get("cache_name"):
            return {
                "status": "error",
                "step": "initial_caching",
                "message": "Failed to create initial cache"
            }
            
        # Test 2: Try to find similar conversation
        similar_message = "How's the weather?"
        response2 = await chat_service.get_response(similar_message)
        
        # Test 3: Retrieve cache directly
        cached_conv = await embedding_service.get_cached_conversation(response1["cache_name"])
        
        return {
            "status": "success",
            "tests": {
                "initial_cache": {
                    "message": test_message,
                    "response": response1["response"],
                    "cache_name": response1["cache_name"],
                    "has_embedding": response1["has_embedding"]
                },
                "similar_query": {
                    "message": similar_message,
                    "response": response2["response"],
                    "used_cache": response2["used_cache"],
                    "has_embedding": response2["has_embedding"]
                },
                "cache_retrieval": {
                    "cache_found": cached_conv is not None,
                    "cached_text": cached_conv["text"] if cached_conv else None
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cache testing failed: {str(e)}"
        }

@app.get("/cache-stats")
async def get_cache_stats():
    """Get statistics about the cache"""
    try:
        caches = await embedding_service.list_caches()
        
        return {
            "status": "success",
            "stats": {
                "total_caches": len(caches),
                "memory_embeddings": len(embedding_service.embedding_storage),
                "memory_conversations": len(embedding_service.cache_storage),
                "cache_details": caches
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get cache stats: {str(e)}"
        }

@app.get("/test-conversation-cache")
async def test_conversation_cache():
    """Test conversation caching with multiple messages"""
    try:
        # Test sequence of messages
        test_sequence = [
            "তোমার নাম কি?",  # What's your name?
            "বাংলাদেশের রাজধানী কি?",  # What's the capital of Bangladesh?
            "ঢাকা সম্পর্কে কিছু বলো"  # Tell me about Dhaka
        ]
        
        results = []
        for message in test_sequence:
            # Send message and get response
            response = await chat_service.get_response(message)
            
            # Get cache details if available
            cache_details = None
            if response.get("cache_name"):
                cache_details = await embedding_service.get_cached_conversation(response["cache_name"])
            
            results.append({
                "user_message": message,
                "bot_response": response.get("response"),
                "cache_name": response.get("cache_name"),
                "has_embedding": response.get("has_embedding", False),
                "used_cache": response.get("used_cache", False),
                "cached_text": cache_details.get("text") if cache_details else None,
                "cache_created": cache_details.get("create_time") if cache_details else None,
                "cache_expires": cache_details.get("expire_time") if cache_details else None
            })
        
        # Try to find similar conversations
        similar_query = "ঢাকা শহর সম্পর্কে জানতে চাই"  # I want to know about Dhaka city
        similar_response = await chat_service.get_response(similar_query)
        
        return {
            "status": "success",
            "conversation_tests": results,
            "similarity_test": {
                "query": similar_query,
                "response": similar_response.get("response"),
                "used_cache": similar_response.get("used_cache", False)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Test failed: {str(e)}"
        }

@app.get("/all-conversations")
async def get_all_conversations():
    """Get all cached conversations with full details"""
    try:
        # Get all caches
        caches = await embedding_service.list_caches()
        logger.info(f"Found {len(caches)} caches")
        
        # Get full conversation details for each cache
        conversations = []
        for cache in caches:
            cache_id = cache["name"]
            logger.info(f"Processing cache: {cache_id}")
            
            conversation = await embedding_service.get_cached_conversation(cache_id)
            
            if conversation:
                # Parse the conversation text to separate user and bot messages
                conv_text = conversation.get("text", "")
                logger.debug(f"Conversation text: {conv_text}")
                
                messages = []
                for line in conv_text.split("\n"):
                    if line.startswith("User: "):
                        messages.append({
                            "role": "user",
                            "content": line[6:],
                            "timestamp": conversation.get("create_time")
                        })
                    elif line.startswith("Bot: "):
                        messages.append({
                            "role": "bot",
                            "content": line[5:],
                            "timestamp": conversation.get("create_time")
                        })
                
                conversations.append({
                    "cache_id": cache_id,
                    "display_name": cache["display_name"],
                    "created_at": cache["create_time"],
                    "expires_at": cache["expire_time"],
                    "messages": messages,
                    "has_embedding": cache_id in embedding_service.embedding_storage
                })
                logger.info(f"Added conversation with {len(messages)} messages")
            else:
                logger.warning(f"No conversation found for cache: {cache_id}")
        
        # Sort conversations by creation time (newest first)
        conversations.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "status": "success",
            "total_conversations": len(conversations),
            "conversations": conversations
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        return {
            "status": "error",
            "message": f"Failed to get conversations: {str(e)}"
        }

@app.get("/conversation-history", response_class=HTMLResponse)
async def conversation_history_page(request: Request):
    """Render conversation history page"""
    return templates.TemplateResponse("conversation_history.html", {"request": request})

@app.post("/check-banglish")
async def check_banglish(text: str = Form(...)):
    """Check and suggest corrections for Banglish text"""
    try:
        suggestions = await chat_service.banglish_service.get_suggestions(text)
        return {
            "status": "success",
            "suggestions": suggestions
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
