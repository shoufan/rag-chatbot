# RAG-Powered Chatbot

## Introduction

This project is a Retrieval-Augmented Generation (RAG) powered chatbot designed to answer questions based on the contents of a PDF document. The chatbot is built using LangChain, IBM Watsonx LLM, and Streamlit for the front-end interface. It uses a policy booklet PDF to generate responses to user queries.

## How the Dataset Was Constructed

To create a robust and comprehensive dataset, I carefully analyzed the provided PDF document to extract meaningful query-response pairs. The goal was to ensure that the dataset covers a diverse range of topics and scenarios mentioned in the document. Here’s the step-by-step process:

1. **Thorough Document Analysis**: I read through the entire policy booklet to understand the different sections and types of information it contains.
2. **Identification of Key Topics**: Key topics such as claims process, coverage details, exclusions, and special conditions were identified.
3. **Query Formulation**: For each key topic, I formulated multiple queries that a user might ask. These queries were designed to cover different aspects and details of the topic.
4. **Response Extraction**: Corresponding responses were extracted directly from the text, ensuring they were accurate and contextually relevant.
5. **Diversity and Balance**: Care was taken to ensure that the queries were not concentrated on a specific section or type of query. This balanced approach helps in evaluating the chatbot’s performance comprehensively.

## Choice of Evaluation Metrics

To evaluate the performance of the chatbot, I selected cosine similarity as the primary metric. Here’s why:

1. **Cosine Similarity**: Measures the cosine of the angle between two vectors in a multi-dimensional space. It is particularly useful for evaluating text similarity as it focuses on the orientation (contextual similarity) rather than magnitude (exact word match).

### Why Cosine Similarity Instead of Traditional Metrics

1. **Textual Nature**: The traditional metrics like accuracy, precision, recall, and F1 score are more suitable for categorical data where exact matches are expected. However, in the context of a chatbot, the responses are textual and may not always match the ground truth exactly word for word.
2. **Contextual Understanding**: Cosine similarity allows us to measure the contextual similarity between the predicted response and the ground truth. This is crucial for a chatbot as it ensures that even if the exact words differ, the meaning conveyed is similar.
3. **Flexibility in Responses**: Using cosine similarity accommodates variations in phrasing and sentence structure, which are common in natural language. This leads to a more robust evaluation of the chatbot’s performance.

By using cosine similarity, we can better assess the chatbot's ability to generate contextually relevant responses, which is more reflective of its real-world performance.

## Why This is a Comprehensive Dataset

I believe this dataset is comprehensive for several reasons:

1. **Diverse Query Types**: The dataset includes queries ranging from simple factual questions to more complex, scenario-based questions. This variety ensures the model is tested on different types of information retrieval.
2. **Coverage of All Sections**: Every major section of the document is represented in the dataset. This helps in evaluating the model’s ability to understand and retrieve information from all parts of the document.
3. **Balanced Approach**: By ensuring the queries are not concentrated on any specific section or type, the dataset provides a balanced evaluation, highlighting both strengths and weaknesses of the model.
4. **Realistic Scenarios**: The queries are designed to mimic real-world questions that users might ask, making the evaluation more relevant and practical.

## Efforts to Improve Accuracy

To improve the accuracy of the chatbot, several strategies were employed:

1. **Normalization of Responses**: Both predicted and ground truth responses were normalized to a common format. This included removing extra spaces, converting to lowercase, and ensuring consistent phrasing.
2. **Parallel Processing**: Implemented parallel processing to speed up the evaluation process, making it more efficient to test and refine the model.
3. **Debugging and Analysis**: Regularly printed and analyzed predicted responses and ground truth responses to identify and address discrepancies. This iterative process helped in fine-tuning the model’s performance.
4. **Use of Vectorization**: Employed vectorization techniques to compare key phrases and content rather than exact matches. This approach helped in evaluating the core information and understanding of the model more effectively.

By implementing these strategies, the model's accuracy and overall performance were continuously monitored and improved, leading to a more robust and reliable chatbot.

## Instructions to Run the Code

To run the RAG-powered chatbot, follow these steps:

### Prerequisites

Ensure you have Python 3.7 or later installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Setup

1. **Clone the Repository**: Clone this repository to your local machine.
    ```bash
    git clone https://github.com/shoufan/rag-chatbot.git
    cd rag-chatbot
    ```

2. **Create a Virtual Environment**: It’s recommended to create a virtual environment to manage dependencies.
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. **Install Dependencies**: Install the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4. **Add PDF Document**: Ensure the `policy-booklet.pdf` file is in the same directory as `app.py`.

5. **Add Dataset**: Ensure the `dataset.csv` file is in the same directory as `app.py`. This file should contain the query-response pairs for evaluation.

### Running the Application

Run the Streamlit application using the following command:
  ```bash
  streamlit run app.py
  ```
