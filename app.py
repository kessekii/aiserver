import sys
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import VitsModel, AutoTokenizer, AutoProcessor, BarkModel
# Load model directly
import numpy as np
from datasets import load_dataset
import scipy
from flask import Flask, request, jsonify
import soundfile as sf
from PIL import Image
import json
import base64
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
import json
import os
import io
import random

import sys
# Adds the other directory to your python path.
sys.path.append("./crop/")
sys.path.append("./crop/main/")
sys.path.append("./crop/main/facecrop/")
from crop.main.facecrop import FaceCrop


def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))
def encode_audio_to_base64(file_path):
    with open(file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    base64_encoded_data = base64.b64encode(audio_data)
    base64_encoded_str = base64_encoded_data.decode('utf-8')
    return base64_encoded_str
def without_keys(d, keys):
   return {x: d[x] for x in d if x not in keys}
def combine_images(base64_image1, base64_image2):
    # Convert base64 strings to images
    img1 = base64_to_image(base64_image1)
    img2 = base64_to_image(base64_image2)

    # Calculate combined dimensions
    combined_width = img1.width + img2.width
    combined_height = max(img1.height, img2.height)

    # Create a new image with the combined dimensions
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the two images side by side
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (img1.width, 0))

    # Convert the combined image to base64
    buffered = io.BytesIO()
    combined_image.save(buffered, format="JPEG")
    combined_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return combined_base64
def image_to_base64(image_path):
    # Open the image file
    with open(image_path, "rb") as image_file:
        # Read the image data
        image_data = image_file.read()
        
        # Encode the image data to base64
        base64_encoded_data = base64.b64encode(image_data)
        
        # Convert the base64 bytes to a string
        base64_encoded_str = base64_encoded_data.decode('utf-8')
        
        return base64_encoded_str
def encode_audio_array_to_base64(data, rate):
    # Create an in-memory bytes buffer
    byte_io = io.BytesIO()
    
    # Write the audio data to the buffer as a WAV file
    scipy.io.wavfile.write(byte_io, rate=rate, data=data)
    
    # Get the byte data from the buffer
    byte_data = byte_io.getvalue()
    
    # Encode the byte data to base64
    base64_encoded_data = base64.b64encode(byte_data)
    
    # Convert the base64 bytes to a string
    base64_encoded_str = base64_encoded_data.decode('utf-8')
    
    return base64_encoded_str
def encode_audio_from_url_to_base64(url):
    # Fetch the audio file from the URL
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    
    # Read the content of the audio file
    audio_data = response.content
    
    # Encode the binary data to base64
    base64_encoded_data = base64.b64encode(audio_data)
    
    # Convert the base64 bytes to a string
    base64_encoded_str = base64_encoded_data.decode('utf-8')
    
    return base64_encoded_str
def check_key_in_json(data, key):
    if key in data:
        return True
    for k, v in data.items():
        if isinstance(v, dict):
            if check_key_in_json(v, key):
                return True
    return False
def get_stream(url):
    s = requests.Session()

    with s.get(url, headers=None, stream=True, ) as resp:
        for line in resp.iter_lines():
            if line:
               
                decoded_str = line.decode('utf-8')
                json_str = decoded_str[len('data: '):]
                
                parsed = json.loads(json_str)
                if 'output' in parsed:
                  
                    return parsed["output"]["data"][0]['url']
                
application = Flask(__name__)

application.wsgi_app = ProxyFix(
    application.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
voiceTokenizeren = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
voiceModelen = VitsModel.from_pretrained("facebook/mms-tts-eng").to("cuda")
voiceTokenizerru = AutoTokenizer.from_pretrained("facebook/mms-tts-rus")
voiceModelru = VitsModel.from_pretrained("facebook/mms-tts-rus").to('cuda')

model.eval()
voiceModelen.eval()
voiceModelru.eval()

@application.route('/facecrop', methods=['POST'])
def facecrop():
    data = request.json
    image_data = data.get('image')
    
    

    
         
    aurus = base64.b64decode(image_data)
    pred = io.BytesIO(aurus)
    image = Image.open(pred)
    image.save('./temp/temp.jpg')
    # parameters = {"width": "60", "height": "60", "width_asy": "0", "height_asy": "0", "tag": "", "folder_option": False, "single_face_option": True}
    face_crop = FaceCrop(width= 60, height= 60, width_asy= 0, height_asy= 0, tag= "")
    # face_crop.__init__(**parameters)
    image=  face_crop.crop_save('./temp/temp.jpg','./temp', './tempout', preview=True)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    print(image)
    img_bytes = buffered.getvalue()
    
    payload = { "image": base64.b64encode(img_bytes).decode('utf-8')}   

    return jsonify(payload)

@application.route('/voice', methods=['POST'])
def voice():
    data = request.json
    theme = data.get('theme')
    name = data.get('name')
    answer = data.get('answer')
   
    language = data.get('language')
    answer = str(answer).lower()
    
    db = {}

    
   
    file_path = os.path.join('.', 'data', 'phraseDataBase.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        db = json.load(file)
  
    if (answer == 'no' and 'no' in db[language][theme]):
        return jsonify({"audio":'null'})
       
    text =  random.choice(db[language][theme][answer]) + str(name)
    
    


# Set pad_token_id
    if (language == 'ru'):
        inputs = voiceTokenizerru(text=str(text), return_tensors="pt").to('cuda')
        with torch.no_grad():

            outputs = voiceModelru(**inputs).waveform
            
            audio_array = outputs.cpu().numpy().squeeze()

            encodedAudio = encode_audio_array_to_base64(audio_array ,rate=voiceModelru.config.sampling_rate)
    else:
        inputs = voiceTokenizeren(text=str(text), return_tensors="pt", ).to('cuda')
        with torch.no_grad():

            outputs = voiceModelen(**inputs).waveform
            audio_array = outputs.cpu().numpy().squeeze()

            encodedAudio = encode_audio_array_to_base64(audio_array ,rate=voiceModelen.config.sampling_rate)

    
    
    
    return jsonify({"audio":encodedAudio})

@application.after_request
def apply_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response

@application.route('/themes')
def themes():
    db={}
    file_path = os.path.join('.', 'data', 'phraseDataBase.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        db = json.load(file)
    result = []
    for key in db['en'].keys(): 
        result.append(key)
        
    result.remove('facerecognition')

    return result

@application.route('/')
def home():
    return 'hello world'

@application.route('/.well-known/pki-validation/57A56D36E3FB3A2146E985A35CBF01AD.txt')
def vakidat():
    return "FC4E11B7A7227194DA7F8D0843E135C75266D953DFFC80E567AD5809D7CDA9D0\nssl.com\n20240627"
   
@application.route('/process', methods=['POST'])
def process_image():
    data = request.json
    
    theme = data.get('theme')
    image_data = data.get('image')
    language = data.get('language')
    db = {}
    # f = open('./data/phraseDataBase.json')
   
    # returns JSON object as 
    # a dictionary
    
    file_path = os.path.join('.', 'data', 'phraseDataBase.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        db = json.load(file)
   
    text =  db[language][theme]['question']
    # image_to_base64('./images/')
    try:
        
        
        aurus = base64.b64decode(image_data)
        pred = io.BytesIO(aurus)
        image = Image.open(pred) 

    except Exception as e:
        application.logger.error(f"Error decoding image: {e}")
        return jsonify({'error': str(e)}), 400
    print(str(text))
    inputs = processor(image, str(text), return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    # image.show()  
    answer  = processor.decode(out[0], skip_special_tokens=True)
    return jsonify({'answer': answer })

@application.route('/compare', methods=['POST'])
def compare():
    data = request.json
    
    image_data = data.get('image')
    
    name = data.get('name')
    db = {}
    # f = open('./data/phraseDataBase.json')
   
    # returns JSON object as 
    # a dictionary
    ppldb = {}
   
    file_path = os.path.join('.', 'data', 'phraseDataBase.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        db = json.load(file)
    file_path = os.path.join('.', 'data', 'peopleDataBase.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        ppldb = json.load(file)
    
    question =  db['en']['facerecognition']['question']
    secindImage =  image_to_base64(str('./data/images/'+str(name).lower()+'.jpg'))
    combined = combine_images(image_data,secindImage )
    try:
        
        
        aurus = base64.b64decode(combined)
        pred = io.BytesIO(aurus)
        image = Image.open(pred) 

    except Exception as e:
        application.logger.error(f"Error decoding image: {e}")
        return jsonify({'error': str(e)}), 400
    print(question)

    inputs = processor(image, question, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs)
    # image.show()  
    answer  = processor.decode(out[0], skip_special_tokens=True)
    return jsonify({'answer': answer })

