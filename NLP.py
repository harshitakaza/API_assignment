#NLP tasks
#1.Question answering
!pip install docx2txt
from transformers import pipeline
import io
from google.colab import files
import docx2txt
def extract_text_from_docx(docx_path):
    """
    Extracts text from a Word document.
    Parameters:
    docx_path (str): Path to the Word document.
    Returns:
    str: Extracted text from the document."""
    return docx2txt.process(docx_path)
def ask_question(question, context=None):
    """
Function to ask a question using a Hugging Face model.
    Parameters:
    question (str): The question you want to ask.
    context (str, optional): Additional context for better answering the question.
    Returns:
    str: The response from the model. """
    try:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad",device=0)
        response = qa_pipeline({
            'question': question,
            'context': context if context else ""
        })
        return response['answer']
    except Exception as e:
        return f"Error: {str(e)}"
def chatbot(context):
    """
    Simple chatbot functionality to answer user questions based on provided context.
    Parameters:
    context (str): The context for answering questions.
    """
    print("Hello! I am your chatbot. Ask me anything, or type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        answer = ask_question(user_input, context)
        print("Bot:", answer)

if __name__ == "__main__":
    print("Please upload a Word document.")
    uploaded = files.upload()
    if uploaded:
        docx_path = list(uploaded.keys())[0]
        context = extract_text_from_docx(docx_path)
        chatbot(context)
    else:
        print("No document uploaded. Exiting.")

#2.Summarization
import pandas as pd
from google.colab import files
from google.colab import drive
drive.mount('/content/drive/')
path = "/content/drive/My\ Drive/Colab Notebooks/email_thread_details.csv"
email_threads_details = pd.read_csv('/content/drive/My Drive/Colab Notebooks/email_thread_details.csv')
email_threads_details
email_threads_summaries = pd.read_csv('/content/drive/My Drive/Colab Notebooks/email_thread_summaries.csv')
email_threads_summaries
combined_data = pd.merge(email_threads_details, email_threads_summaries, on='thread_id')
# instantiate a text summarization pipeline
summarizer = pipeline('summarization')

# Iterate over each email thread
email_content = combined_data['body'].iloc[0]
# Generate a summary using the text summarization pipeline
summary = summarizer(email_content, max_length=10000, min_length=150, do_sample=False)
# Print the original email content and the generated summary
print("Original Email Content:")
print(email_content)
print("\nGenerated Summary:")
print(summary[0]['summary_text'])
print("-"*30)
#3.Text classification based on sentiment analysis
# instantiate a text classification pipeline
classifier = pipeline('sentiment-analysis')

# Iterate over each email thread
email_content = combined_data['summary'].iloc[10]
# Generate a classifier result using the classification pipeline
classifier_result = classifier(email_content,max_length=512)
# Print the original email contentn and the classifier rsult
print("Original Email Content--------------:")
print(email_content)
print("\n\n Calssifer result:----------------")
print(classifier_result)
#4.Token classification
# instantiate a token classification pipeline
token_classifier = pipeline('ner')

# Iterate over each email thread
email_content = combined_data['body'].iloc[0]
# Generate a token classifier result using the classification pipeline
token_classifier_result = token_classifier(email_content)
# Print the original email contentn and the token classifier rsult
print("Original Email Content--------------:")
print(email_content)
print("\n\n Calssifer result:----------------")
print(token_classifier_result)
#5.Translation
translation_content = combined_data['summary'][0]
translation_content
# instantiate a text classification pipeline
translator = pipeline('translation_xx_to_yy', model='unicamp-dl/translation-en-pt-t5')

# Iterate over each email thread
translation_content = combined_data['summary'][0]
# Generate a classifier result using the classification pipeline
translation_result = translator(translation_content,max_length = 1000000)
# Print the original email contentn and the classifier rsult
print("Original  Content--------------:")
print(translation_content)
print("\n\n translation result:----------------")
print(translation_result)

