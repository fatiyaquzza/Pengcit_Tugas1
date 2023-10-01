from ssl import AlertDescription
from flask import Flask, redirect, render_template, request, url_for
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('ai.html')


if __name__ == "__main__":
    app.run(debug=True)
