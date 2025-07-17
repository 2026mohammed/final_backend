import requests

file_path = 'test2.jpg'  # ضع هنا اسم الصورة
url = 'http://127.0.0.1:8000/predict'

with open(file_path, 'rb') as f:
    files = {'file': (file_path, f, 'image/jpeg')}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
