from huggingface_hub import InferenceClient
from dotenv import load_dotenv

client = InferenceClient(
    provider="nebius",
    api_key=HF_TOKEN,
)

def generate_gemma_table_only(results_list):
    if not results_list:
        return "No analysis in the last minute."
    # Tablo başlıkları ve satırlar
    result_str = "| No | Class | Symmetry | Edge Density | Spot |\n|---|---|---|---|---|\n"
    for i, res in enumerate(results_list, 1):
        result_str += f"| {i} | {res['predicted_class']} | {res['symmetry_score']:.2f} | {res['edge_density']:.3f} | {res['spot_count']} |\n"
    prompt = (
        "Show only the following Markdown table and nothing else. "
        "DO NOT write any explanation, summary, or extra text. "
        "Return ONLY the table below. "
        "Keep the answer under 40 tokens.\n"
        f"{result_str}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    completion = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=messages,
        max_tokens=40  # token sayısı KISITLANDI!
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    sample_results = [
        {"predicted_class": "clean", "symmetry_score": 0.8, "edge_density": 0.01, "spot_count": 0},
        {"predicted_class": "color_defect_detection", "symmetry_score": 0.5, "edge_density": 0.09, "spot_count": 3}
    ]
    table = generate_gemma_table_only(sample_results)
    print("---- TABLO BAŞLADI ----")
    print(table)
    print("---- TABLO BİTTİ ----")
