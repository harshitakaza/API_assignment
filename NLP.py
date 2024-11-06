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
