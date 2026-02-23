import nltk
import os

nltk_data_dir = os.path.join(os.getcwd(), "data", "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

print(f"NLTK data dir: {nltk_data_dir}")

try:
    nltk.download("punkt", download_dir=nltk_data_dir)
    nltk.download("punkt_tab", download_dir=nltk_data_dir)
    print("Downloaded punkt and punkt_tab")
except Exception as e:
    print(f"Download failed: {e}")

try:
    nltk.data.find("tokenizers/punkt")
    print("Found punkt")
    nltk.data.find("tokenizers/punkt_tab")
    print("Found punkt_tab")
except LookupError as e:
    print(f"Lookup failed: {e}")

# Test tokenization
try:
    tokens = nltk.word_tokenize("Hello world")
    print(f"Tokenization works: {tokens}")
except Exception as e:
    print(f"Tokenization failed: {e}")
