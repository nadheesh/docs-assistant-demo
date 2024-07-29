import os
import gradio as gr
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "gpt-3.5-turbo"
PERSIST_DIRECTORY = "docs.db"
MAX_TOKENS = 3000

# Initialize components
encodings = tiktoken.encoding_for_model(MODEL_NAME)
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())
chat = ChatOpenAI(model_name=MODEL_NAME)

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


def get_doc_prompt(results):
    doc_prompts = []
    for result in results:
        content = result.page_content.replace("(../assets", "(file/assets").replace("{.cInlineImage-full}", "")
        doc_prompts.append(f"Document: {{content: {content}, metadata:{result.metadata}}}")
    return "\n".join(doc_prompts)


def create_message(prompt, history):
    messages = [HumanMessage(content=prompt)]
    prompt_size = len(encodings.encode(prompt))

    for user_msg, ai_msg in reversed(history):
        new_size = prompt_size + len(encodings.encode(user_msg + " " + ai_msg))
        if new_size > MAX_TOKENS:
            break
        messages.extend([AIMessage(content=ai_msg), HumanMessage(content=user_msg)])
        prompt_size = new_size

    return list(reversed(messages))


def generate_response(message, history):
    try:
        results = db.similarity_search(message)
        doc_content = get_doc_prompt(results)
        chat_prompt = PROMPT_TEMPLATE.format(doc_content=doc_content, user_question=message)

        print(f"Prompt length: {len(encodings.encode(chat_prompt))}")

        reply = ""
        for chunk in chat.stream(create_message(chat_prompt, history)):
            reply += chunk.content
            yield reply
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        yield "I'm sorry, but an error occurred while processing your request. Please try again later."


if __name__ == "__main__":
    gr.ChatInterface(generate_response).launch(allowed_paths=["assets/"])