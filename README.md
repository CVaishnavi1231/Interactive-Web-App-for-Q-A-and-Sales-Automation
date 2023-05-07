# LANGCHAIN BASED APP


This is a project that uses leverages Langchain and a Vector DB to to perform various natural language processing (NLP) tasks.
The app is implemented using Streamlit, a popular Python library for building interactive web applications.

## Tasks

The app includes the following three main functions:

### Task 1: Q&A With PDF File

This function allows users to upload a PDF file and then interact with a conversational AI chatbot to ask questions about the content of the PDF file.

### Task 2: Customer Sentiment Analysis

This function performs sentiment analysis on customer reviews, and displays the sentiment scores and sentiment words for each product.The sentiment score is calculated by analyzing the sentiment of each review and assigning a score of 1 for positive, -1 for negative, or 0 for neutral to each product, which is then summed up for all reviews of that product.


### Task 3: Competitor Analysis

This usecase performs competitor analysis on comparing product features of the users brand with other brands of the same product using OpenAI's language model.
#### NOTE : For Task 3 please upload a CSV file containing the product description of your brand in the first row, followed by the descriptions of other brands for the same product in subsequent rows.




## Requirements

The project requires several dependencies as listed in the requirements.txt file 

* langchain==0.0.161
* openai==0.27.6
* pypdf==3.8.1
* streamlit==1.22.0
* streamlit-chat==0.0.2.2
* faiss-cpu==1.7.4
* tiktoken==0.3.3
* chromadb==0.3.21

##  Installation and Usage

#### Step 1: 
Install the required packages and start the application

```ruby
!chmod +x run_app.sh
!run_app.sh
```

#### Step 2: 
The above command outputs a url. Click on the url to navigate to start the Streamlit server and open the application in your default web browser.

#### Step 3: 
Press on the "CLICK TO CONTINUE" box to navigate to the Langchain Based App.

#### Step 4: 
Interact with the App by choosing the usecase you want.

#### NOTE: For the task 2 and 3 please upload the provided csv file. For Task 2, upload Brand_Specifications.csv file and for Task 3 upload the Customer_Reviews.csv file .
