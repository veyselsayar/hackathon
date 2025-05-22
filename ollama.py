import requests

# Test için analiz sonucunu örnekle (API'den veya manuel gir)
results = {
    "bounding_box": [100, 100, 300, 300],
    "symmetry_score": 0.13,
    "edge_density": 0.017,
    "spot_count": 3,
    "predicted_class": "integrity_check",
    "probabilities": [0.15, 0.79, 0.06]
}


def generate_llama_summary(result_dict):
    summary_prompt = f"""
    Bir medikal göz pedi kalite kontrolü yaptım. Sonuçlar şöyle:
    - Sınıf: {result_dict['predicted_class']}
    - Simetri skoru: {result_dict['symmetry_score']}
    - Kenar yoğunluğu: {result_dict['edge_density']}
    - Leke sayısı: {result_dict['spot_count']}
    - Sınıf olasılıkları: {result_dict['probabilities']}

    Bu sonuçlara göre, kullanıcının kolay anlayacağı şekilde detaylı bir açıklama ve öneri yaz.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",  # Ollama'nın default endpoint'i
        json={"model": "llama3", "prompt": summary_prompt, "stream": False}
    )
    output = response.json()
    return output.get("response", "")


if __name__ == "__main__":
    summary = generate_llama_summary(results)
    print("Llama Cevabı:\n")
    print(summary)
