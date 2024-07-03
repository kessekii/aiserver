import io
import requests
import base64
from PIL import Image


    
encoded_image = Image.open("C:/Users/hamme/Downloads/test2.jpg").convert('RGB')
encoded_image = Image.open("C:/Users/hamme/Downloads/test2.jpg").convert('RGB')
encoded_image = Image.open("C:/Users/hamme/Downloads/test2.jpg").convert('RGB')
encoded_image = Image.open("C:/Users/hamme/Downloads/test2.jpg").convert('RGB')

buffered = io.BytesIO()
encoded_image.save(buffered, format="JPEG")

img_bytes = buffered.getvalue()
print(img_bytes)
payload = { "image": base64.b64encode(img_bytes).decode('utf-8'), "text": "Are they the same?"}   

response = requests.post("http://85.65.185.254:8000/process", json=payload)

