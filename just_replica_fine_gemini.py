import re
from datetime import datetime, timedelta
import streamlit as st
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from utility import validate_input,extract_date_from_query
from instructions import instructions_text,prompt_template_chatpdf_bookAppointment2,instructions_text3_gemini
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from dotenv import load_dotenv
import os
import sys
import json


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: Google API Key not found in environment.")
else:
    genai.configure(api_key=api_key)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print(embeddings)
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        print("FAISS index created and saved successfully.")
    except Exception as e:
        print(f"Error during FAISS index creation: {e}")



def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. \
    If the answer is not in the context, say "answer is not available in the context".
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# def chat_pdf(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     #st.write("Reply: ", response["output_text"])
#     return response["output_text"]

def chat_pdf(user_query):
    # Check if a PDF has been uploaded in the current session
    print("user_query: ",user_query)

    if not st.session_state.get("pdf_uploaded", False):
        return "No PDF has been uploaded. Please upload a PDF file before querying it."


    #sys.exit("Exiting the program")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return f"Error loading the FAISS index: {e}. Please upload a PDF file before querying it."

    docs = new_db.similarity_search(user_query)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_query}, return_only_outputs=True)
    return response["output_text"]


def book_appointment(query):
    # Extract information from the query
    
    print("\ncheck query:",query)
    #sys.exit("Exiting the program")
    user_query_json = json.loads(query)
    name = user_query_json['Name']
    email = user_query_json['Email']
    phone = user_query_json['PhoneNumber']
    date = user_query_json['Date']

    # Check if we have all necessary information
    missing_info = []
    if not name:
        missing_info.append("name")
    if not email:
        missing_info.append("email")
    if not phone:
        missing_info.append("phone number")
    if not date:
        missing_info.append("appointment date")

    if missing_info:
        return f"I couldn't find the following information in your query: {', '.join(missing_info)}. " \
               f"Please provide all necessary details to book an appointment."

    # Validate email and phone
    if not validate_input(email, "email"):
        return "The email address provided is not valid. Please try again with a valid email."
    if not validate_input(phone, "phone"):
        return "The phone number provided is not valid. Please try again with a valid phone number."

    # Book the appointment
    appointment_date = extract_date_from_query(date)
    return f"Appointment booked for {appointment_date} for {name} (Email: {email}, Phone: {phone})."

def langchain_agent(user_query):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    

    tools = [
        Tool(
            name="ChatPDF",
            func=chat_pdf,
            description="Use this to query the PDF for summaries or specific information about the document content."
        ),
        Tool(
            name="BookAppointment",
            func=book_appointment,
            description="Use this to book appointments. The query should contain name, email, phone number, and desired appointment date/time."
        ),
    ]

    prompt_template = PromptTemplate.from_template(instructions_text3_gemini)

    # Create the react agent using the prompt, llm, and tools
    agent = create_react_agent(llm=llm, prompt=prompt_template, tools=tools)

    # Create a memory object to store conversation history
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create an agent executor from the agent and tools
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        #memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        #max_iterations=1
    )

    max_attempts = 1
    attempts = 0

    while attempts < max_attempts:
        try:
            result = agent_executor.invoke({"input": user_query})
            output = result.get('output', '')

            # Check if the output contains the missing information message
            if "I couldn't find the following information in your query" in output:
                return output  # Return the missing information message directly

            return output  # Return any other output

        except Exception as e:
            print(f"Error during agent execution: {e}")
            attempts += 1
            if attempts >= max_attempts:
                return "Agent stopped due to iteration limit or time limit."

    return "Maximum attempts reached. Please try rephrasing your query."

    #result = agent_executor.invoke({"input": user_query})
    #return result['output']

def main():
    st.set_page_config("Chat PDF and Appointment Booking")
    st.header("Chat with PDF and Book Appointments using Gemini💁")

    #user_query = st.text_input("Ask a question, summarize PDF, or book an appointment")
    user_query = st.text_area("Ask a question, summarize PDF, or book an appointment")

    if st.button("Send") and user_query:
        with st.spinner("Processing..."):
            response = langchain_agent(user_query)
            print("response",response)
            st.write("Reply: ", response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.pdf_uploaded = True  # Set the session state
                    st.success("Done")

if __name__ == "__main__":
    main()