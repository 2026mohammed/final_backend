import os
import gdown

url = "https://drive.google.com/uc?id=1AZuoRgRVVY1_OWJDlXr1IUtNT-xjyGLL"
output_path = "model/plant_disease_model.h5"

os.makedirs("model", exist_ok=True)

if not os.path.exists(output_path):
    print("Downloading model...")
    gdown.download(url, output_path, quiet=False)
else:
    print("Model already exists.")
