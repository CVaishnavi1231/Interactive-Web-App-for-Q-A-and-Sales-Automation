import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from streamlit_chat import message
import panel as pn
import tempfile
import json
import csv
import os



def function_1():

    """
    function_1() implements a Streamlit app that allows users to ask questions related to a PDF document,
    and get answers from a ConversationalRetrievalChain model, which uses OpenAI's GPT-3.5 language model 
    and FAISS vectorstore to retrieve the answers from the PDF document.

    """

    st.info("INFO: This usecase allows users to upload a PDF file and then interact with a conversational AI chatbot to ask questions about the content of the PDF file. ")

    # Input field for the OpenAI API key
    user_api_key = st.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

    # Set the OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = user_api_key

    

    # File uploader for a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:

        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load the PDF document into memory
        loader = PyPDFLoader(file_path=tmp_file_path)
        documents = loader.load()

        # Split the text into chunks for easier processing
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Generate embeddings for the text
        embeddings = OpenAIEmbeddings()
        # Create a vector store and search index from the documents and embeddings
        docsearch =  FAISS.from_documents(documents, embeddings)

       
        # Create a conversational retrieval chain from a chat-based model and the vector store
        chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
                                                                      retriever=docsearch.as_retriever())

        # Function to handle user queries and return the chatbot's response
        def conversational_chat(query):
        
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            
            return result["answer"]
    
        # Initialize session state variables for chat history, generated messages, and past user inputs
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name ]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hi"]
            
        #container for the chat history
        response_container = st.container()

        #container for the user's text input
        container = st.container()

        # form to take user's input and display the chat history and generated responses
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:

                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            # display the chat history and generated responses
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


def function_2():

    """
    function_2() implements a Streamlit app to perform sentiment analysis on customer reviews stored
    in a CSV file using OpenAI's GPT-3.5 Turbo model.The sentiment scores and sentiment words are then 
    displayed for each product mentioned in the reviews.
    
    """

    st.info("INFO: This usecase performs sentiment analysis on customer reviews, and displays the sentiment scores and sentiment words for each product.")

 
    user_api_key = st.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

    os.environ["OPENAI_API_KEY"] = user_api_key

    # Allows user to upload a CSV file with customer reviews
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    # Perform with sentiment analysis only if a file is uploaded
    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # read reviews from CSV file
        reviews = []
        with open(tmp_file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                reviews.append(row)

        # Set up the OpenAI LLMChain for sentiment analysis
        llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

        #streamlit view component
        text_review = 'product'

        #1 prompt template
        template = """
        Please act as a machine learning model trained to perform a supervised learning task, 
        to extract the sentiment of a review.

        Give your answer writing a Json evaluating the sentiment field between the dollar sign, the value must be printed without dollar sign.
        The value of sentiment must be "positive"  or "negative", otherwise if the text is not valuable write "null". Give the sentiment words as sentiment_words also

        Example:

        field 1 named :
        text_review with value: {text_review}
        field 2 named :
        sentiment with value: $sentiment$
        filed 3 named: sentiment_words 
        keyword with value: 

        Review text: '''{text_review}'''

        """

        # Create a PromptTemplate with the above template
        prompt = PromptTemplate(template=template, input_variables=["text_review"])

        # initialize a LLMChain instance with the prompt template and the OpenAI instance
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # Initialize variables for storing sentiment scores and words for each product
        question = text_review
        d = {}
        sent_word = {}
        rating = {'positive':1 , 'negative':-1, 'null':0}

        # Loop through each customer review and perform sentiment analysis
        if prompt:
            for r in reviews:
                prod = r['product_name']
                re = r['review_text']
                response = llm_chain.run(re)
                data = json.loads(response)
                temp = data["sentiment"]
                temp = temp.strip("$")

                # Add the sentiment score for the product to the dictionary
                if prod in d:
                    d[prod] += rating[temp]
                else:
                    d[prod] = rating[temp]
                
                # Add the sentiment words for the product to the dictionary
                if prod in sent_word:
                    sent_word[prod].extend(data["sentiment_words"])
                else:
                    sent_word[prod] = data["sentiment_words"]

        st.info("INFO: The sentiment score is calculated by analyzing the sentiment of each review and assigning a score of 1 for positive, -1 for negative, or 0 for neutral to each product, which is then summed up for all reviews of that product.")
         # Display the sentiment scores for each product
        sentence = "The sentiment scores for the products are: \n"

        for product, score in d.items():
            sentence = sentence + "{}: {}, ".format(product, score)

        sentence = sentence[:-2] + ".\n" + "The sentiment words for each product is displayed below"

        st.success(sentence)
        for product, words in sent_word.items():
          formatted_words = ", ".join(words)
          st.write(product + ":")
          st.write("[" + formatted_words + "]")

def function_3():
   
    """
    function_3() implements a Streamlit app to perform competitor analysis on Brand Specifications stored
    in a CSV file using OpenAI's GPT-3.5 Turbo model. It reads a CSV file containing product descriptions 
    and compares the features of the product of the users brand with other brands. 
   
    """

    st.info("INFO: This usecase performs competitor analysis on comparing product features of the users brand with other brands of the same product using OpenAI's language model.")
    user_api_key = st.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

    os.environ["OPENAI_API_KEY"] = user_api_key

    st.info("NOTE: Please upload a CSV file containing the product description of your brand in the first row, followed by the descriptions of other brands for the same product in subsequent rows. ")
    
    # Allows user to upload a CSV file with Brand specifications
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    # Perform with competitor analysis only if a file is uploaded
    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # read reviews from CSV file
        reviews = []
        with open(tmp_file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                reviews.append(row)
        

        # Set up the OpenAI LLMChain for sentiment analysis
        llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        template = """
        Please act as a machine learning model trained for perform a supervised learning task, 
        for comparing the description of the same product from different brands, where you want to see 
        how the product from Brand_A fares with all other brands
        Give your answer as bullet points evaluating the comparison field between the dollar sign, the value must be printed without dollar sign.
        The comparison must be between Brand_A and all other brands as product features what features Brand_A lacks and what features Brand_A has. 
        Example:

        $comparison$


        Review text: '''{text_review}'''

        """

        # Create a PromptTemplate with the above template
        prompt = PromptTemplate(template=template, input_variables=["text_review"])

        # initialize a LLMChain instance with the prompt template and the OpenAI instance
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # generate responses for the reviews
        response = llm_chain.run(reviews)
        st.success(response)

def main():

    st.title("Langchain Based App")
  
    function_options = ["No Selection", "Q&A With PDF File", "Customer Sentiment Analysis", "Competitor Analysis"]
    choice = st.selectbox("Select a function", function_options)
    st.header(choice)
    if choice == "Q&A With PDF File":
        function_1()
    elif choice == "Customer Sentiment Analysis":
        function_2()
    elif choice == "Competitor Analysis":
        function_3()
    elif choice == "No Selection":
        st.info ("Please make a selection")

if __name__ == "__main__":
    main()
