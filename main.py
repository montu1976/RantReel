import os
import json
import glob
import requests
from flask import Flask, request, jsonify, render_template

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

# -------------------------
# LOAD SEMANTIC MODEL
# -------------------------
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# LOAD DATASET
# -------------------------
DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

def load_dataset():
    results = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("instruction") or obj.get("input") or obj.get("prompt")
                    resp = obj.get("response") or obj.get("completion")
                    if text and resp:
                        results.append({"input": text, "response": resp})
                except:
                    pass
    return results


# -------------------------
# SEMANTIC MATCHING
# -------------------------
def match_dataset(user_text, dataset):
    user_words = set(user_text.lower().split())
    best = None
    best_score = 0

    for item in dataset:
        ex_words = set(item["input"].lower().split())
        score = len(user_words & ex_words)
        if score > best_score:
            best = item
            best_score = score

    if best_score >= 2:
        return best

    return None



# -------------------------
# HUGGINGFACE FALLBACK
# -------------------------
def try_huggingface(prompt):
    try:
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            print("NO HF TOKEN FOUND")
            return None

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 120}
        }

        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"

        res = requests.post(url, headers=headers, json=payload, timeout=20)
        data = res.json()

        print("HF Raw Response:", data)

        # Format: [{"generated_text": "..."}]
        if isinstance(data, list) and len(data) > 0:
            if "generated_text" in data[0]:
                return data[0]["generated_text"]

        # Format: {"generated_text": "..."}
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]

        # Direct string
        if isinstance(data, str):
            return data

        # HF error
        if "error" in data:
            print("HF error:", data["error"])
            return None

        return None

    except Exception as e:
        print("HF Exception:", e)
        return None


# -------------------------
# MAIN CHAT ROUTE
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    # Load dataset
    dataset = load_dataset()

    # 1️⃣ Try dataset (semantic match)
    match = match_dataset(user_message, dataset)
    if match:
        return jsonify({"response": match["response"], "source": "dataset"})

    # 2️⃣ Try HuggingFace
    hf_reply = try_huggingface(user_message)
    if hf_reply:
        return jsonify({"response": hf_reply, "source": "huggingface"})

    # 3️⃣ Final fallback
    return jsonify({
        "response": "I'm here for you. Tell me more.",
        "source": "fallback"
    })


# -------------------------
# LIST DATASETS
# -------------------------
@app.route("/datasets")
def list_data():
    files = os.listdir(DATA_DIR)
    total = len(files)
    return jsonify({"files": files, "total_files": total})


# -------------------------
# HOME PAGE → WEB CHAT UI
# -------------------------
@app.route("/")
def home():
    return render_template("chat.html")


# -------------------------
# RUN FOR RENDER
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
