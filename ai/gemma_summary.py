from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv


HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(
    api_key=HF_TOKEN
)

def generate_gemma_summary(results_list):
    if not results_list:
        return "No analysis in the last minute."
    total = len(results_list)
    faulty = sum(1 for r in results_list if r['predicted_class'] != "clean")

    # Prompt'ta sadece sayıyı yaz ve ardından kısa bir yorum iste
    prompt = (
        f"{total} products were inspected. "
        f"{faulty} of them are faulty (not 'clean'). "
        "State only the number of faulty products and write a very short comment. Do not write more than 2-3 sentences."
            "Your response must be very short and in Turkish."
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
        max_tokens=35  # Çok kısa yanıt için düşük tut
    )
    return completion.choices[0].message.content.strip()

# Test
if __name__ == "__main__":
    test_results = [
        {"predicted_class": "clean", "symmetry_score": 0.8, "edge_density": 0.01, "spot_count": 0},
        {"predicted_class": "color_defect_detection", "symmetry_score": 0.5, "edge_density": 0.09, "spot_count": 3},
        {"predicted_class": "clean", "symmetry_score": 0.9, "edge_density": 0.015, "spot_count": 0},
        {"predicted_class": "integrity_check", "symmetry_score": 0.4, "edge_density": 0.08, "spot_count": 2}
    ]

