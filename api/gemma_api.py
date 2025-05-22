from flask import Flask, jsonify
from ai.collect_results import get_results_and_clear
from ai.gemma_summary import generate_gemma_summary

app = Flask(__name__)

@app.route("/gemma_summary", methods=["GET"])
def gemma_summary():
    results = get_results_and_clear(older_than_seconds=65)
    summary = generate_gemma_summary(results)
    return jsonify({
        "summary": summary,
        "result_count": len(results)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5052, debug=True)
