import re
from datetime import datetime, timedelta
import streamlit as st
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # Updated import for OpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from utility import validate_input, extract_date_from_query
from instructions import instructions_text, prompt_template_chatpdf_bookAppointment2, instructions_text3_openai
from langchain.vectorstores import FAISS  # Import from the original library
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # Updated for OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # Changed to OpenAI API Key
if not api_key:
    print("Error: OpenAI API Key not found in environment.")
else:
    # Set the OpenAI API Key here if necessary, depending on the library's implementation
    pass  

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
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Use OpenAI embeddings
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
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)  # Update to OpenAI model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def chat_pdf(query):
    print("\ncheck query:",query)
    if not st.session_state.get("pdf_uploaded", False):
        return "No PDF has been uploaded. Please upload a PDF file before querying it."

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Use OpenAI embeddings
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return f"Error loading the FAISS index: {e}. Please upload a PDF file before querying it."

    docs = new_db.similarity_search(query)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": query}, return_only_outputs=True)
    return response["output_text"]

def book_appointment(query):
    print("\ncheck query:",query)
    user_query_json = json.loads(query)
    name = user_query_json['Name']
    email = user_query_json['Email']
    phone = user_query_json['PhoneNumber']
    date = user_query_json['Date']

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

    if not validate_input(email, "email"):
        return "The email address provided is not valid. Please try again with a valid email."
    if not validate_input(phone, "phone"):
        return "The phone number provided is not valid. Please try again with a valid phone number."

    appointment_date = extract_date_from_query(date)
    return f"Appointment booked for {appointment_date} for {name} (Email: {email}, Phone: {phone})."

def langchain_agent(user_query):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)  # Update to OpenAI model

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

    prompt_template = PromptTemplate.from_template(instructions_text3_openai)

    agent = create_react_agent(llm=llm, prompt=prompt_template, tools=tools)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    max_attempts = 3
    attempts = 0


    while attempts < max_attempts:
        try:
            result = agent_executor.invoke({"input": user_query})
            output = result.get('output', '')

            if "I couldn't find the following information in your query" in output:
                return output  # Return the missing information message directly
            

            return output  # Return any other output

        except Exception as e:
            print(f"Error during agent execution: {e}")
            attempts += 1
            if attempts >= max_attempts:
                return "Agent stopped due to iteration limit or time limit."

    return "Maximum attempts reached. Please try rephrasing your query."

def main():
    st.set_page_config("Chat PDF and Appointment Booking")
    st.header("Chat with PDF and Book Appointments using OpenAI 💁")

    user_query = st.text_area("Ask a question, summarize PDF, or book an appointment")
    print("\ntaking query here:",user_query)

    if st.button("Send") and user_query:
        with st.spinner("Processing..."):
            response = langchain_agent(user_query)
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
