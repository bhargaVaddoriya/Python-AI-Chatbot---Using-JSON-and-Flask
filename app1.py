import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify, render_template
from markupsafe import Markup  # Import Markup from markupsafe package
import requests
import random

app = Flask(__name__, template_folder='template')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', output_attentions=False)

# Function to fetch data from API using provided API key
def fetch_data_from_api(search_query, subscription_key):
    api_url = f'https://api.nhs.uk/conditions/{search_query}?subscription-key={subscription_key}'
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')
        return None

# Function to process the API data and generate HTML content
def process_data(data):
    if not data:
        return 'Error: Failed to fetch data from the API'

    has_part_html = ''
    first_iteration = True
    if isinstance(data['mainEntityOfPage'], list):
        for part in data['mainEntityOfPage']:
            if not first_iteration:
                if part.get('headline') and part.get('description'):
                    has_part_html += f'<h1>{part["headline"]}</h1><h3>{part["description"]}</h3>'
            else:
                first_iteration = False

            if isinstance(part.get('hasPart'), list):
                for nested_part in part['hasPart']:
                    if nested_part.get('text'):
                        has_part_html += f'<p>{nested_part["text"]}</p>'
                    if nested_part.get('@type') == 'ImageObject' and nested_part.get('url'):
                        has_part_html += f'<img src="{nested_part["url"]}" alt="Image"><p>{nested_part["name"]}</p>'
    else:
        if data['mainEntityOfPage'].get('headline') and data['mainEntityOfPage'].get('description'):
            has_part_html = f'<h1>{data["mainEntityOfPage"]["headline"]}</h1><h3>{data["mainEntityOfPage"]["description"]}</h3>'

    return f'<h1>{data["description"]}</h1>{has_part_html}'

# Custom dataset for greetings and basic Q&A
custom_responses = {
    'hi': ['Hello! How can I assist you today?', 'Hi there! How can I help you?'],
    'hii': ['Hello! How can I assist you today?', 'Hi there! How can I help you?'],
    'hiii': ['Hi there! How can I help you?', 'Hello! How can I assist you today?'],
    'hello': ['Hi there! How can I help you?', 'Hello! How can I assist you today?'],
    'how are you': ['I am just a chatbot, but thanks for asking!', 'I\'m doing well, thanks!'],
    'what is your name': ['I am a chatbot created to provide information about various health conditions.', 'I\'m a chatbot designed to help you with health-related questions.'],
    'thank you': ['You\'re welcome! If you have any more questions, feel free to ask.', 'No problem! Feel free to ask anything else.'],
    'mumbai':['mumbai is a best city.'],
    
    # Add more question-answer pairs here
}


def handle_custom_responses(question):
    question_lower = question.lower()
    if question_lower in custom_responses:
        return random.choice(custom_responses[question_lower])
    else:
        return None


@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['message']

    # Check for custom responses
    custom_response = handle_custom_responses(question)
    if custom_response:
        return jsonify({'response': Markup(custom_response)})

    # Call the function to fetch data from the API
    subscription_key = '9f41a40c13df4aeeb293366233c59ec4'
    api_data = fetch_data_from_api(question, subscription_key)
    
    # Process the API data and generate HTML content
    if api_data:
        html_response = process_data(api_data)
        return jsonify({'response': Markup(html_response)})  # Wrap HTML response in Markup
    else:
        return jsonify({'response': 'Error: Failed to fetch data from the API'})

if __name__ == "__main__":
    app.run(debug=True, threaded=False)