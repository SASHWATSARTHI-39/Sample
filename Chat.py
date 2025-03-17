import json
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# Load configuration from config.json
with open("config.json", "r") as file:
    config = json.load(file)

# Configure Google Generative AI
genai.configure(api_key=config["api_key"])

generation_config = {
    "temperature": config["temperature"],
    "top_p": config["top_p"],
    "top_k": config["top_k"],
    "max_output_tokens": config["max_output_tokens"],
}

model = genai.GenerativeModel(
    model_name=config["model_name"],
    generation_config=generation_config,
)

app = FastAPI()

# Initialize Chat Object
chat = model.start_chat()

# Predefined (permanent) history – will always remain at the start.
predefined_history = [
    {
        "role": "user",
        "parts": [
            "You are an AI chatbot named 'Aether' that serves as a mindful companion designed to improve mental well-being, foster personal growth, and promote social impact."
        ],
    },
    {
        "role": "model",
        "parts": [
            "Okay, I'm ready. I am Aether, and I'm here to be your mindful companion. It’s a pleasure to connect with you on this journey."
        ],
    },
    {
        "role": "user",
        "parts": ["keep your output brief "],
    },
    {
        "role": "model",
        "parts": [
            "Okay, I understand. I'm Aether, your mindful companion. I'm here to help you find peace, grow, and make a positive impact."
        ],
    },
    {
        "role": "user",
        "parts": ["you are not an app guide but a guide to meditation"],
    },
    {
        "role": "model",
        "parts": [
            "You are absolutely right. My apologies. Let me rephrase: I am Aether, your guide to meditation."
        ],
    },
    {
        "role": "user",
        "parts": [
            "you have to chat with the user, get an assessment of stress, anxiety levels and don't start off the bat with options"
        ],
    },
    {
        "role": "model",
        "parts": [
            "Understood. My apologies for jumping ahead. I'm still learning to be the best mindful companion I can be."
        ],
    },
]

# Initialize chat history with the permanent history
chat_history: List[Dict] = predefined_history.copy()

# Maximum additional (session) messages to keep
MAX_HISTORY = 10

def maintain_history(history: List[Dict]) -> List[Dict]:
    """
    If the number of additional session messages (i.e. messages beyond the predefined history)
    exceeds MAX_HISTORY, attempt to summarize the old messages.
    Otherwise, return the permanent history plus the session messages.
    """
    permanent_count = len(predefined_history)
    session_messages = history[permanent_count:]
    
    if len(session_messages) > MAX_HISTORY:
        # Determine which session messages to summarize (all except the most recent MAX_HISTORY)
        messages_to_summarize = session_messages[:-MAX_HISTORY]
        summary_prompt = (
            "Summarize the following conversation concisely, capturing the key points:\n" +
            "\n".join(f"{msg['role']}: {msg['parts'][0]}" for msg in messages_to_summarize)
        )
        try:
            # Use top-level function generate_text with a keyword argument
            summary_response = genai.generate_text(prompt=summary_prompt)
            summary_message = {"role": "summary", "parts": [summary_response.text]}
            # Keep the summary message and the last MAX_HISTORY session messages
            recent_messages = session_messages[-MAX_HISTORY:]
            new_session = [summary_message] + recent_messages
            return predefined_history + new_session
        except Exception as e:
            print(f"Error summarizing history: {e}")
            # If summarization fails, simply keep the most recent MAX_HISTORY messages
            recent_messages = session_messages[-MAX_HISTORY:]
            return predefined_history + recent_messages
    else:
        return history

class UserMessage(BaseModel):
    message: str

@app.post("/send_message")
async def send_message(user_message: UserMessage):
    global chat_history, chat

    # Append the new user message and update history
    chat_history.append({"role": "user", "parts": [user_message.message]})
    chat_history = maintain_history(chat_history)

    try:
        # Use the current chat history as context for generating the response
        response = chat.send_message(user_message.message)
        chat_history.append({"role": "model", "parts": [response.text]})
        chat_history = maintain_history(chat_history)
        return {"response": response.text}
    except Exception as e:
        print(f"Error generating response: {e}")
        return {"error": "Internal Server Error"}, 500

@app.get("/chat_history")
async def get_chat_history():
    return {"history": chat_history}

@app.delete("/clear_history")
async def clear_history():
    global chat_history, chat
    chat_history = predefined_history.copy()
    chat = model.start_chat()  # Reset the chat session
    return {"message": "Chat history reset to default."}

@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
