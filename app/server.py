import os
import json
import numpy as np
import faiss
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model, FAISS index, and metadata at startup
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('./faiss_index/index.faiss')
with open('./faiss_index/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Load Phi-3 Mini LLM and tokenizer
phi3_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
phi3_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

app = Flask(__name__)

def retrieve(query, top_k=5):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), top_k)
    results = []
    for idx in I[0]:
        meta = metadata[idx]
        results.append(meta)
    return results

def generate_answer(query, contexts):
    context_texts = "\n".join([c['text'] for c in contexts])
    prompt = f"Context:\n{context_texts}\n\nQuestion: {query}\nAnswer:"
    inputs = phi3_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = phi3_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            pad_token_id=phi3_tokenizer.eos_token_id
        )
    answer = phi3_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    top_chunks = retrieve(user_query, top_k=5)
    answer = generate_answer(user_query, top_chunks)
    citations = [
        {"text": c["text"], "source": c["file"]}
        for c in top_chunks
    ]
    return jsonify({
        "answer": answer,
        "citations": citations
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
