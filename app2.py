import random
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__, template_folder='template')


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

    # Look for the question in the dataset
    for item in qa_dataset['questions']:
        if item['question'].lower() == question.lower():
            # Select a random answer from the list of answers for the question
            answer = random.choice(item['answers'])
            return jsonify({'response': answer})

    # If the question is not found in the dataset, respond with a default message
    default_response = "Sorry, I couldn't find an answer to that question."
    return jsonify({'response': default_response})
        



if __name__ == "__main__":
    app.run(debug=True, threaded=False)
