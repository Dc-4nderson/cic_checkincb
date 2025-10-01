"""
Flask UI for CIC checkins RAG chatbot.

Features:

Run: flask --app flask_app.py run
"""
from __future__ import annotations
import os
from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify
from data_handling import load_checkins, upsert_checkins_to_pinecone, get_embeddings
from rag import build_context_from_query, generate_answer_from_context, query_pinecone_by_vector

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")

# Theme URL
BLACK_BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"

# Templates
CHECKINS_TEMPLATE = '''
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>CIC Checkins — RAG Chatbot</title>
    <link href="{{ theme_url }}" rel="stylesheet">
    <style>
        body { background-color: #000 !important; }
        .bg-black { background-color: #000 !important; }
        .border-secondary { border-color: #444 !important; }
    </style>
</head>
<body class='bg-black text-light'>
<div class='container py-4'>
    <nav class="mb-4">
        <a href="{{ url_for('checkins') }}" class="btn btn-outline-light me-2">Checkins</a>
        <a href="{{ url_for('chatbot') }}" class="btn btn-outline-light">Chatbot</a>
    </nav>
    <h1 class='mb-4'>CIC Checkins</h1>
    <div class='row'>
        <div class='col-md-8'>
            <h4>Recent Checkins</h4>
            <ul class='list-group mb-3'>
            {% for c in checkins %}
                <li class='list-group-item bg-black text-light border-secondary'><strong>{{c.timestamp}}</strong> — {{c.checkin}}</li>
            {% endfor %}
            </ul>
        </div>
        <div class='col-md-4'>
            <h4>Upsert to Pinecone</h4>
            <form method='post' action='{{ url_for("upsert") }}'>
                <button type='submit' class='btn btn-primary'>Upsert all checkins</button>
            </form>
        </div>
    </div>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class='mt-3'>
        {% for msg in messages %}
          <div class='alert alert-warning'>{{ msg }}</div>
        {% endfor %}
        </div>
      {% endif %}
    {% endwith %}
</div>
</body>
</html>
'''

CHATBOT_TEMPLATE = '''
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>CIC Chatbot</title>
    <link href="{{ theme_url }}" rel="stylesheet">
    <style>
        body { background-color: #000 !important; }
        .bg-black { background-color: #000 !important; }
        .border-secondary { border-color: #444 !important; }
    </style>
</head>
<body class='bg-black text-light'>
<div class='container py-4'>
    <nav class="mb-4">
        <a href="{{ url_for('checkins') }}" class="btn btn-outline-light me-2">Checkins</a>
        <a href="{{ url_for('chatbot') }}" class="btn btn-outline-light">Chatbot</a>
    </nav>
    <h1 class='mb-4'>CIC Chatbot</h1>
    <div class='row'>
        <div class='col-md-8'>
            <form method='post' action='{{ url_for("chatbot") }}'>
                <div class='mb-2'>
                    <label>Question</label>
                    <input type='text' name='question' class='form-control bg-black text-light border-secondary' value='{{ question|default("") }}' required>
                </div>
                <div class='mb-2'>
                    <label>Context hits (topK)</label>
                    <input type='number' name='topk' class='form-control bg-black text-light border-secondary' value='{{ topk|default(5) }}' min='1' max='10'>
                </div>
                <button type='submit' class='btn btn-success'>Ask</button>
            </form>
            {% if answer %}
            <div class='mt-4'>
                <h5>Answer</h5>
                <div class='alert alert-info bg-black text-light border-secondary'>{{ answer }}</div>
                <h6>Context used</h6>
                <pre class='bg-black text-light p-2 border border-secondary'>{{ context }}</pre>
            </div>
            {% endif %}
        </div>
    </div>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class='mt-3'>
        {% for msg in messages %}
          <div class='alert alert-warning'>{{ msg }}</div>
        {% endfor %}
        </div>
      {% endif %}
    {% endwith %}
</div>
</body>
</html>
'''

def get_env(name, default=None):
    return os.getenv(name, default)

@app.route("/")
def home():
    return redirect(url_for("checkins"))

@app.route("/checkins", methods=["GET"])
def checkins():
    checkins = load_checkins("my_checkins.json")[::-1]
    return render_template_string(CHECKINS_TEMPLATE,
        checkins=checkins,
        pinecone_api_url=get_env("PINECONE_API_URL", ""),
        pinecone_api_key=get_env("PINECONE_API_KEY", ""),
        embedding_method=get_env("EMBEDDING_METHOD", "sbert"),
        openai_api_key=get_env("OPENAI_API_KEY", ""),
        theme_url=BLACK_BOOTSTRAP
    )

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    answer, context = None, None
    question = None
    topk = 5
    pinecone_api_url = os.getenv("PINECONE_API_URL")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if request.method == "POST":
        question = request.form.get("question")
        topk = int(request.form.get("topk", 5))
        if not pinecone_api_url or not pinecone_api_key or not openai_api_key:
            flash("Pinecone/OpenAI API config missing in environment variables or .env file.", "warning")
        else:
            try:
                from data_handling import _openai_embeddings
                q_emb = _openai_embeddings([question], openai_api_key)[0]
                pinecone_resp = query_pinecone_by_vector(pinecone_api_url, pinecone_api_key, q_emb, topK=topk)
                context = build_context_from_query(pinecone_resp)
                answer = generate_answer_from_context(question, context, openai_api_key)
            except Exception as e:
                flash(f"Query failed: {e}", "warning")
    return render_template_string(CHATBOT_TEMPLATE,
        answer=answer,
        context=context,
        question=question,
        topk=topk,
        theme_url=BLACK_BOOTSTRAP
    )

@app.route("/upsert", methods=["POST"])
def upsert():
    checkins = load_checkins("my_checkins.json")
    pinecone_api_url = os.getenv("PINECONE_API_URL")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not pinecone_api_url or not pinecone_api_key:
        flash("Pinecone API URL and API Key must be set in environment variables or .env file.", "warning")
        return redirect(url_for("checkins"))
    try:
        res = upsert_checkins_to_pinecone(pinecone_api_url, pinecone_api_key, None, checkins, openai_api_key=openai_api_key)
        flash(f"Upserted {res.get('upserted',0)} vectors", "info")
    except Exception as e:
        flash(f"Upsert failed: {e}", "warning")
    return redirect(url_for("checkins"))

if __name__ == "__main__":
    app.run(debug=True)
