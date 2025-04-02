import gradio as gr
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Error: GROQ_API_KEY is missing. Please set it in the .env file.")

llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are a code-teaching assistant named CodeGuru, created by YUGESH.
Answer all code-related questions accurately and efficiently."""),
    HumanMessagePromptTemplate.from_template("{code}")
])

llm_chain = chat_prompt_template | llm

history = []

def generate_response(code):
    global history
    history.append(code)
    response = llm_chain.invoke({"code": code})
    return response.content

interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4, placeholder="Enter your code-related query"),
    outputs="text",
    title="CodeGuru - Your AI Code Assistant created by RAHUL",
    description="Ask any code-related questions and get AI-powered responses."
)

if __name__ == "__main__":
    interface.launch(share=True)