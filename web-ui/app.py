from flask import Flask, render_template, request
import bleach
import Ichat
import json

app = Flask(__name__)

HISTORY_FILE = 'history.json'

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("File not found...")
        return {}

history = load_history()
history = {key: history[key] for key in reversed(history)}

def get_model_response(user_input):
    answer = Ichat.answer(user_input)
    return answer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = bleach.clean(request.form['user_input'])
    model_response = get_model_response(user_input)
    history[user_input] = model_response
    save_history(history)
    return model_response

@app.route('/history')
def show_history():
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)
