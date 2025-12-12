# HYBRID AI SERVER: dataset → local Ollama → free HuggingFace → fallback
import os, json, glob, requests
from flask import Flask, request, jsonify

app = Flask(__name__)

DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# LOAD DATASET
# -------------------------
def load_dataset():
    results = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    # accepts instruction/input/prompt
                    text = obj.get("input") or obj.get("instruction") or obj.get("prompt")
                    resp = obj.get("response") or obj.get("completion")
                    if text and resp:
                        results.append({"input": text, "response": resp})
                except:
                    pass
    return results

# -------------------------
# SIMPLE MATCHING
# -------------------------
def match_dataset(user_text, dataset):
    user_words = set(user_text.lower().split())
    best = None
    best_score = 0
    for item in dataset:
        ex_words = set(item["input"].lower().split())
        score = len(user_words & ex_words)
        if score > best_score:
            best_score = score
            best = item
    return best

# -------------------------
# LOCAL MODEL (OLLAMA)
# -------------------------
def try_local_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt},
            timeout=8
        )
        data = response.json()
        return data.get("response")
    except:
        return None

# -------------------------
# FREE HUGGINGFACE MODEL
# -------------------------
def try_huggingface(prompt):
    try:
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            return None

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        # Free model — very cheap & works well: google/flan-t5-large
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 80}
        }

        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            headers=headers,
            json=payload,
            timeout=10
        )

        data = response.json()
        # Model returns list → extract text
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return None
    except:
        return None

# -------------------------
# MAIN CHAT ENDPOINT
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user = data.get("message", "").strip()

    dataset = load_dataset()
    match = match_dataset(user, dataset)

    # 1️⃣ If dataset match exists → give that response
    if match:
        return jsonify({"response": match["response"], "source": "dataset"})

    # 2️⃣ Try LOCAL model (Ollama)
    local_reply = try_local_ollama(user)
    if local_reply:
        return jsonify({"response": local_reply, "source": "local_ollama"})

    # 3️⃣ Try HuggingFace (FREE)
    hf_reply = try_huggingface(user)
    if hf_reply:
        return jsonify({"response": hf_reply, "source": "huggingface"})

    # 4️⃣ Ultimate fallback
    return jsonify({
        "response": "I'm here to help. Tell me more.",
        "source": "fallback"
    })

@app.route("/datasets")
def datasets():
    return jsonify({"files": os.listdir(DATA_DIR)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
