from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from chat_service import ChatService
from banglish_service import BanglishCorrector

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

chat_service = ChatService()
banglish_corrector = BanglishCorrector()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    response = await chat_service.get_response(message)
    return response

@app.post("/correct-banglish")
async def correct_banglish(text: str = Form(...)):
    corrected_text = await banglish_corrector.correct_text(text)
    return {"corrected_text": corrected_text}
