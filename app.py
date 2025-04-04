import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("RAG Chatbot")

# File upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    response = requests.post(f"{API_URL}/upload_pdf/", files=files)
    if response.status_code == 200:
        st.sidebar.success(response.json().get("message", "PDF uploaded successfully!"))
    else:
        st.sidebar.error("Failed to upload PDF.")

# Chat input & response area with history
st.header("Ask a question")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Enter your question:")
if st.button("Get Answer") and user_query:
    response = requests.post(f"{API_URL}/chat/", data={"user_query": user_query})
    # st.write("API Response:", response.text)  

    if response.status_code == 200:
        answer_data = response.json()
        answer = answer_data.get("answer", "No answer found.")  
        st.session_state.chat_history.append((user_query, answer))
        st.write(answer)
    else:
        st.error("Failed to retrieve response.")

# Display chat history
st.subheader("Chat History")
for query, answer in st.session_state.chat_history:
    st.write(f"**Q:** {query}")
    st.write(f"**A:** {answer}")
    st.write("---")

# Clear database button
if st.sidebar.button("Clear Database"):
    response = requests.post(f"{API_URL}/clear_database/")
    if response.status_code == 200:
        st.sidebar.success("Database cleared successfully!")
    else:
        st.sidebar.error("Failed to clear database.")
