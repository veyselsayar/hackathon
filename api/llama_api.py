from flask import Flask, jsonify
import requests

# Modül yolu ayarı (ai klasörünü bulmak için)

from ai.collect_results import get_results_and_clear
# Flask app
app = Flask(__name__)

def generate_llama_summary(results_list):
    if not results_list:
        return "Son 1 dakikada analiz yok."
    result_str = ""
    for i, res in enumerate(results_list, 1):
        result_str += (
            f"{i}. Sınıf: {res['predicted_class']}, Simetri: {res['symmetry_score']:.2f}, "
            f"Kenar Yoğunluğu: {res['edge_density']:.3f}, Leke: {res['spot_count']}\n"
        )
    prompt = (
        f"Son 1 dakikada yapılan göz pedi analizleri:\n"
        f"{result_str}\n"
        "Bu sonuçlara göre, üretim bandındaki genel kalite durumu nedir? Hataları özetle, tavsiye ver."
    )
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    output = response.json()
    return output.get("response", "")

@app.route("/llama_summary", methods=["GET"])
def llama_summary():
    results = get_results_and_clear(older_than_seconds=65)
    summary = generate_llama_summary(results)
    return jsonify({
        "summary": summary,
        "result_count": len(results)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
