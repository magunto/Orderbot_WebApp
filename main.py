#! /bin/python3
# version claude

from flask import Flask, render_template, request, jsonify
import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

app = Flask(__name__)

# Constants
MAX_CONTEXT_LENGTH = 10
INITIAL_MESSAGE = "Welcome to Pizza Palace. What can I get for you today?"

# Change working directory to the script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)


class ChatBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        self.groq_client = Groq(api_key=self.api_key)
        self.model = "llama-3.2-90b-text-preview"
        self.context: List[Dict[str, str]] = []
        self.load_initial_context()

    def load_initial_context(self):
        try:
            with open("content.txt", "r") as file:
                self.context = [{"role": "system", "content": file.read()}]
        except FileNotFoundError:
            print("Warning: content.txt not found, using empty initial context")
            self.context = []

    def get_completion(self, messages, temperature=0):
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=7000,
                top_p=0.5,
                stream=True,
            )

            return "".join(
                chunk.choices[0].delta.content or ""
                for chunk in completion
                if chunk.choices
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def get_response(self, message: str) -> str:
        self.context.append({"role": "user", "content": message})
        if len(self.context) > MAX_CONTEXT_LENGTH:
            self.context.pop(1)

        response = self.get_completion(self.context)
        self.context.append({"role": "assistant", "content": response})
        return response


chatbot = ChatBot()


@app.route("/")
def home():
    return render_template("chat.html", initial_message=INITIAL_MESSAGE)


@app.route("/send_message", methods=["POST"])
def send_message():
    message = request.json.get("message")
    response = chatbot.get_response(message)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
