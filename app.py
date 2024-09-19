import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__, template_folder='template')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', output_attentions=False)

# Load QA dataset
with open('dataset.json', 'r') as f:
    qa_dataset = json.load(f)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['message']

    # Initialize list to store all answers
    all_answers = []

    # Look for the question in the dataset
    for item in qa_dataset['questions']:
        if item['question'].lower() == question.lower():
            # Append all answers for the question to the list
            all_answers.extend(item['answers'])

    # If answers are found for the question, return them
    if all_answers:
        return jsonify({'response': all_answers})

    # If the question is not found in the dataset, respond with a default message
    default_response = "Sorry, I couldn't find an answer to that question."
    return jsonify({'response': default_response})
        

if __name__ == "__main__":
    app.run(debug=True, threaded=False)
