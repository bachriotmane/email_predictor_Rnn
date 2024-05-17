# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
def url_predictor(url):
    tokenizer = AutoTokenizer.from_pretrained("elftsdmr/malware-url-detect")
    model = AutoModelForSequenceClassification.from_pretrained("elftsdmr/malware-url-detect")

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Test the URLs
    results = pipe(url)

    # Print the results
    # for url, result in zip(urls, results):
    #     print(f"URL: {url}\nPrediction: {result['label']} - Score: {result['score']:.4f}\n")
    return results[0]['label']