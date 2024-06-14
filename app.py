# import langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# pandas for dataset & scikitlearn for model evaluation
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# To speed up model evaluation
import concurrent.futures
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# streamlit for UI dev
import streamlit as st

# watsonx interface
from langchain_ibm import WatsonxLLM

# Create llm using langchain
llm = WatsonxLLM(
    apikey ='yWjJr9MMNhI6QvPfj4HvWfGWbEXtoUUndwTFYeaQ4ghr',
    url= "https://us-south.ml.cloud.ibm.com",
    model_id = 'meta-llama/llama-2-70b-chat',
    params = {
        'decoding-method':'sample',
        'max_new_tokens':200,
        'temperature':0.5
    },
    project_id = '449af03d-2e2e-4cd0-9c6d-aee7cfa66862')

# This function Loads a PDF of your choosing
@st.cache_resource
def load_pdf():
    # Update PDF name to whatever you like
    pdf_name = 'policy-booklet.pdf'
    loaders = [PyPDFLoader(pdf_name)]
    # Create index - aka vector database - aka chromadb
    index = VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    ).from_loaders(loaders)
    # Return the vector database
    return index

# Load PDF up
index = load_pdf()

# Create a Q&A chain using ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=index.vectorstore.as_retriever()
)

# App title
st.title('Ask Kurisu')

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display all historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the prompts
prompt = st.chat_input('Pass Your Prompt here')

# If the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role': 'user', 'content':prompt})

    # Send the prompt to the PDF Q&A Chain
    response = chain.run({"question": prompt, "chat_history": st.session_state.chat_history})
    # Show the LLM response
    st.chat_message('assistant').markdown(response)
    # Store the LLM response in state
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    # Update chat history
    st.session_state.chat_history.append((prompt, response))

# Normalizing function
def normalize_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase and strip leading/trailing spaces
    text = text.strip().lower()
    return text

# Model Evaluation
def evaluate_model():
    df = pd.read_csv('dataset.csv')
    ground_truth = df['response'].tolist()
    predicted_responses = []

    # Use a smaller subset for testing
    subset_size = 10
    subset_queries = df['query'].tolist()[:subset_size]
    ground_truth_subset = ground_truth[:subset_size]

    def get_response(query):
        return chain.run({"question": query, "chat_history": []})

    # Use concurrent futures for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        predicted_responses = list(executor.map(get_response, subset_queries))

    # Normalize responses
    predicted_responses = [normalize_text(resp) for resp in predicted_responses]
    ground_truth_subset = [normalize_text(resp) for resp in ground_truth_subset]

    # Debugging: Print out the predicted responses and ground truth
    st.write("Predicted Responses:")
    st.write(predicted_responses)
    st.write("Ground Truth:")
    st.write(ground_truth_subset)

    # Calculate cosine similarity
    vectorizer = TfidfVectorizer().fit(predicted_responses + ground_truth_subset)
    predicted_vectors = vectorizer.transform(predicted_responses).toarray()
    ground_truth_vectors = vectorizer.transform(ground_truth_subset).toarray()

    similarities = cosine_similarity(predicted_vectors, ground_truth_vectors)

    # Aggregate similarity scores
    avg_similarity = similarities.diagonal().mean()
    avg_similarity_rounded = round(avg_similarity, 3)

    st.write(f"Average Cosine Similarity: {avg_similarity_rounded}")

if st.button('Evaluate Model'):
    evaluate_model()