from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chatbot
from config import settings

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chatbot.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Kothakoli API"} 