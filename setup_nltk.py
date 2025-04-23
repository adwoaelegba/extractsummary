import nltk
import os
import shutil

# Define the path where nltk data is stored
nltk_path = './nltk_data'
punkt_path = os.path.join(nltk_path, 'tokenizers', 'punkt')

# Step 1: Download 'punkt' tokenizer
nltk.download('punkt', download_dir=nltk_path)

# Step 2: Remove unnecessary files (keep only english.pickle)
for file in os.listdir(punkt_path):
    if file != 'english.pickle':
        os.remove(os.path.join(punkt_path, file))

