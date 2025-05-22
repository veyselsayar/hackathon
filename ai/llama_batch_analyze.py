import time
from collect_results import get_results_and_clear
import requests
def generate_llama_summary(results_list):
    if not results_list:
        return "No analysis in the last minute."
    result_str = ""
    for i, res in enumerate(results_list, 1):
        result_str += (
            f"{i}. Class: {res['predicted_class']}, Symmetry: {res['symmetry_score']:.2f}, "
            f"Edge Density: {res['edge_density']:.3f}, Spot: {res['spot_count']}.\n"
        )

    prompt = (
        f"""You are a senior QA expert and technical report writer.
    Below is the Markdown table showing the quality control results of eye pads from the last minute:

    {result_str}

    First, show ONLY the table and write in Turkish Language.
    Then, in maximum 3 very short, clear sentences, summarize only the most important findings.
    - If there are no defects, simply say: 'All products appear intact.'
    - If there are defects, describe ONLY the problem and give one brief suggestion.
    Do NOT add any extra explanation or text outside the table and summary.
    Your response should never be more than 100 tokens.
    """
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    output = response.json()
    return output.get("response", "")

if __name__ == "__main__":
    while True:
        time.sleep(60)
        results = get_results_and_clear(older_than_seconds=100)
        print(f"\n--- Son 1 dakikanın toplu özeti ({len(results)} kayıt): ---")
        if results:
            summary = generate_llama_summary(results)
            print(summary)
        else:
            print("Son 1 dakikada analiz yok.")
