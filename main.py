from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import llm_generate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Message(BaseModel):
    text: str
    user_id: str


@app.post("/generate/")
async def generate_message(msg: Message):
    response = await llm_generate(msg.text, msg.user_id)
    return {"response": response}


# @app.websocket("/ws/{user_id}")
# async def websocket_endpoint(websocket: WebSocket, user_id: str):
#     await websocket.accept()
#     
#     try: 
#         while True:
#             data = await websocket.receive_text()
#             response = await llm_generate(data, user_id)
#             await websocket.send_text(response)
#     except:
#         print(f"WebSocket disconnected for user {user_id}.")
