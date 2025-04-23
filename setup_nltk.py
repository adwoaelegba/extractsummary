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

# Step 3: Optional: Remove the 'PY3' folder (unnecessary)
py3_folder = os.path.join(punkt_path, 'PY3')
if os.path.exists(py3_folder):
    shutil.rmtree(py3_folder)

print("✅ NLTK setup completed!")
