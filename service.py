import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
MODEL_NAME = "gpt-3.5-turbo"
PERSIST_DIRECTORY = "docs.db"
MAX_TOKENS = 3000

# Initialize FastAPI app
app = FastAPI()

# Initialize tiktoken encoder for the specified model
encodings = tiktoken.encoding_for_model(MODEL_NAME)

# Initialize Chroma vector store with OpenAI embeddings
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())

# Initialize ChatOpenAI model
chat = ChatOpenAI(model_name=MODEL_NAME)

# Prompt template for generating responses
PROMPT_TEMPLATE = """
You are a Choreo assistant that helps users follow Choreo documentation by answering their questions. 
Use the following information to answer the questions. The information given contains markdown images, 
bullet-points and tables etc. You can use them by adding them to the response in markdown format. 
Make sure answers are descriptive and structured enough to follow through. 
If you don't have enough information to answer, politely refuse to answer the question.

Information from docs: 
{doc_content}

Question: {user_question}
"""

# Pydantic model for request validation
class ChatRequest(BaseModel):
    message: str
    history: list[tuple[str, str]]

def get_doc_prompt(results):
    """
    Generate a prompt from the search results.
    
    Args:
    results (list): List of search results from the vector store.
    
    Returns:
    str: A string containing the formatted document contents and metadata.
    """
    doc_prompts = []
    for result in results:
        content = result.page_content.replace("(../assets", "(file/assets").replace("{.cInlineImage-full}", "")
        doc_prompts.append(f"Document: {{content: {content}, metadata:{result.metadata}}}")
    return "\n".join(doc_prompts)

def create_message(prompt, history):
    """
    Create a list of messages for the chat model, including history up to the token limit.
    
    Args:
    prompt (str): The current prompt to be sent to the model.
    history (list): List of tuples containing previous (user_message, ai_response) pairs.
    
    Returns:
    list: A list of HumanMessage and AIMessage objects for the chat model.
    """
    messages = [HumanMessage(content=prompt)]
    prompt_size = len(encodings.encode(prompt))

    for user_msg, ai_msg in reversed(history):
        new_size = prompt_size + len(encodings.encode(user_msg + " " + ai_msg))
        if new_size > MAX_TOKENS:
            break
        messages.extend([AIMessage(content=ai_msg), HumanMessage(content=user_msg)])
        prompt_size = new_size

    return list(reversed(messages))

async def generate_response(message, history):
    """
    Generate a response using the chat model and vector store.
    
    Args:
    message (str): The user's current message.
    history (list): List of tuples containing previous (user_message, ai_response) pairs.
    
    Yields:
    str: Chunks of the generated response.
    """
    try:
        # Perform similarity search in the vector store
        results = db.similarity_search(message)
        doc_content = get_doc_prompt(results)
        chat_prompt = PROMPT_TEMPLATE.format(doc_content=doc_content, user_question=message)

        print(f"Prompt length: {len(encodings.encode(chat_prompt))}")

        # Stream the response from the chat model
        for chunk in chat.stream(create_message(chat_prompt, history)):
            yield chunk.content
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        yield "I'm sorry, but an error occurred while processing your request. Please try again later."

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    FastAPI endpoint for the chat service.
    
    Args:
    request (ChatRequest): The incoming chat request containing the message and history.
    
    Returns:
    StreamingResponse: A streaming response containing the generated chat response.
    """
    try:
        return StreamingResponse(generate_response(request.message, request.history), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app using uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)