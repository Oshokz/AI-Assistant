"""
Kola AI: A Language Assistant capable of performing multiple tasks
such as sentiment analysis, translation, summarization, and comment classification.
Additionally, Kola AI can answer general questions using the power of Llama 3.2.
"""

import streamlit as st  # Importing the Streamlit library for UI creation
from langchain_ollama import OllamaLLM  # Importing the OllamaLLM model for language tasks

# Initialize the language model with specific parameters for balanced responses
model = OllamaLLM(
    model="llama3.2",  # Specify the model name (llama3.2)
    temperature=0.1,  # Low temperature for more consistent and focused outputs
    top_p=0.95,  # Limits sampling to top probable responses
    top_k=50,  # Limits to the top 50 most likely words during generation
    max_tokens=200  # Maximum number of tokens to generate in each response
)

# Function to answer general questions
def ask_a_question(question: str) -> str:
    """
    Answer a general question with high accuracy and provide only the most relevant and concise information.
    The answer should be clear and informative.
    Zero-shot task with precise and fact-based output.
    """
    # Creating the prompt for answering the general question
    question_prompt = f"""
    Provide a clear, concise, and factual answer to the following general question. 
    Do not provide any speculative or ambiguous responses. Your answer should be based solely on verifiable information.

    Question: {question}
    Answer:
    """
    # Generate the response based on the prompt
    result = model.generate(prompts=[question_prompt])
    # Return the generated response as a cleaned-up string
    return result.generations[0][0].text.strip()

# Function to translate text
def translate_text(text: str, source_language: str, target_language: str) -> str:
    """
    Translate text from source language to target language.
    Zero-shot task with very strict translation required.
    """
    # Creating the translation prompt with a strict requirement for accuracy
    translation_prompt = f"""
    Translate the following text from {source_language} to {target_language} accurately and without losing any meaning. 
    The translation must be as close to the original meaning as possible.

    Text: {text}
    Translation:
    """
    # Generate the translation response
    result = model.generate(prompts=[translation_prompt])
    # Return the translated text
    return result.generations[0][0].text.strip()

# Function to analyze sentiment and give a percentage of positivity/negativity with advice
def analyze_sentiment(feedback: str) -> str:
    """
    Analyze the sentiment of a given text.
    Sentiment analysis should return a percentage of positive/negative sentiment and advice.
    """
    # Creating the prompt for sentiment analysis with a focus on sentiment percentage
    sentiment_prompt = f"""
    Analyze the sentiment of the following text. Return a percentage of how positive or negative it is.
    If the sentiment is positive, provide advice on maintaining it. If negative, provide advice on how to improve it.
    
    Feedback: {feedback}
    Sentiment Analysis:
    """
    # Generate the sentiment analysis result
    result = model.generate(prompts=[sentiment_prompt])
    
    # Extract the result (assuming the model returns a percentage and advice in a structured manner)
    sentiment_result = result.generations[0][0].text.strip()
    
    # Return the sentiment analysis result with advice
    return sentiment_result


# Function to summarize text
def summarize_text(text: str) -> str:
    """
    Summarize the given text.
    Provide a clear and concise summary with the most important points only.
    """
    # Creating the prompt for text summarization
    summarization_prompt = f"""
    Summarize the following text in one paragraph. Be concise and include only the essential points.
    Text: {text}
    Summary:
    """
    # Generate the summary result
    result = model.generate(prompts=[summarization_prompt])
    # Return the summary text
    return result.generations[0][0].text.strip()

# Function to classify a customer comment
def classify_comment(comment: str) -> str:
    """
    Classify a comment as 'Damaged' or 'Opinion'.
    Classification must strictly choose between 'Damaged' or 'Opinion', no other options.
    """
    # Creating the prompt for comment classification
    classification_prompt = f"""
    Classify the following comment as either 'Damaged' or 'Opinion'. 
    Strictly adhere to the provided options and do not classify under any other category.
    
    Comment: {comment}
    Classification:
    """
    # Generate the classification result
    result = model.generate(prompts=[classification_prompt])
    # Return the classification result
    return result.generations[0][0].text.strip()

# Streamlit UI starts here
st.title("Kola AI - Your Language Assistant")  # Display the app's title at the top
st.markdown("<h5 style='text-align: center;'>Powered by Meta Llama 3.2</h5>", unsafe_allow_html=True)

# Dropdown menu for selecting a task
task = st.selectbox(
    "Select a task:",  # Label for the dropdown menu
    [
        "Select a task...",  # Placeholder option
        "Ask a general question",  # Ask a question option
        "Translate text between languages",  # Translation task
        "Perform Sentiment Analysis",  # Sentiment analysis task
        "Summarize a piece of text",  # Summarization task
        "Classify customer comment as Damaged or Opinion"  # Comment classification task
    ]
)

# Logic for the General Question task
if task == "Ask a general question":  # Check if the user selected the 'Ask a general question' task
    question = st.text_input("Ask a general question to Kola AI:")  # Input box for the user's question
    if st.button("Ask"):  # Button to submit the question
        if question:  # Ensure a question is entered
            with st.spinner("Getting answer..."):  # Show a loading spinner
                answer = ask_a_question(question)  # Call the function to answer the question
            st.success(f"Answer: {answer}")  # Display the answer
        else:
            st.error("Please enter a question.")  # Show error if no question is provided

# Logic for the Translate Text task
elif task == "Translate text between languages":  # Check if the user selected the translation task
    source_lang = st.selectbox(
        "Select source language:",  # Prompt for selecting the source language
        ["English", "French", "Spanish", "German", "Chinese", "Hindi", "Arabic", "Russian"]
    )
    text = st.text_input(f"Type text in {source_lang}:")  # Input box for text in the selected source language
    target_lang = st.selectbox(
        "Select target language:",  # Prompt for selecting the target language
        ["English", "French", "Spanish", "German", "Chinese", "Hindi", "Arabic", "Russian"]
    )
    if st.button("Translate"):  # Button to trigger translation
        if text and source_lang and target_lang:  # Ensure all fields are filled
            with st.spinner("Translating in progress..."):  # Show a spinner while translating
                translation = translate_text(text, source_lang, target_lang)  # Call the translation function
            st.success(f"Translation from {source_lang} to {target_lang}: {translation}")  # Display the translation
        else:
            st.error("Please enter text to translate and select both languages.")  # Error if inputs are missing

if task == "Perform Sentiment Analysis":
    feedback = st.text_area("Enter feedback to analyze:")  # Input box for the feedback
    if st.button("Analyze Sentiment"):  # Button to trigger sentiment analysis
        if feedback:  # Ensure feedback is entered
            with st.spinner("Analyzing sentiment..."):  # Show a spinner while analyzing sentiment
                sentiment = analyze_sentiment(feedback)  # Call the sentiment analysis function
            st.success(f"Sentiment Analysis: {sentiment}")  # Display the sentiment result
        else:
            st.error("Please enter feedback to analyze.")  # Error if no feedback is entered
            
# Logic for the Summarize Text task
elif task == "Summarize a piece of text":  # Check if the user selected the summarization task
    text_to_summarize = st.text_area("Enter the text to summarize:")  # Input box for the text to summarize
    if st.button("Summarize"):  # Button to trigger summarization
        if text_to_summarize:  # Ensure text is entered
            with st.spinner("Summarizing in progress..."):  # Show a spinner while summarizing
                summary = summarize_text(text_to_summarize)  # Call the summarization function
            st.success(f"Summary: {summary}")  # Display the summary
        else:
            st.error("Please enter text to summarize.")  # Error if no text is entered

# Logic for the Comment Classification task
elif task == "Classify customer comment as Damaged or Opinion":  # Check if the user selected the comment classification task
    customer_comment = st.text_input("Enter a customer's comment about a returned item:")  # Input box for the comment
    if st.button("Classify Comment"):  # Button to trigger comment classification
        if customer_comment:  # Ensure comment is entered
            with st.spinner("Classifying in progress..."):  # Show a spinner while classifying
                classification = classify_comment(customer_comment)  # Call the classification function
            st.success(f"Classification: {classification}")  # Display the classification result
        else:
            st.error("Please enter a comment to classify.")  # Error if no comment is entered
