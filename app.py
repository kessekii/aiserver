import sys
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import VitsModel, AutoTokenizer, AutoProcessor, BarkModel , WhisperForConditionalGeneration, WhisperProcessor
# Load image_clsf_model directly
import numpy as np

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
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq
import whisper

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
# from compreface import CompreFace
# from compreface.service import RecognitionService
# from compreface.collections import FaceCollection
# from compreface.collections.face_collections import Subjects

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


# compre_face: CompreFace = CompreFace('http://localhost','8050')

# recognition_service: RecognitionService = compre_face.init_face_recognition('2df50b47-0d98-4363-9899-cf683f184de0')
image_clsf_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
image_clsf_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to('cuda')
voice_tokenizer_en = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
voice_model_en = VitsModel.from_pretrained("facebook/mms-tts-eng").to('cuda')
voice_tokenizer_ru = AutoTokenizer.from_pretrained("facebook/mms-tts-rus")
voice_model_ru = VitsModel.from_pretrained("facebook/mms-tts-rus").to('cuda')
voice_tokenizer_he = AutoTokenizer.from_pretrained("facebook/mms-tts-heb")
voice_model_he = VitsModel.from_pretrained("facebook/mms-tts-heb").to('cuda')
# llm_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
# llm_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium").to('cuda')
llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct").to('cuda')
translation_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt" )
translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to('cuda')
speech_recog_model = whisper.load_model("base").to('cuda')
image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to("cuda")

image_clsf_model.eval()
# voice_model_en.eval()
# voice_model_ru.eval()
# voice_model_he.eval()
# translation_model.eval()
@application.route('/facecrop', methods=['POST'])
def facecrop():
    data = request.json
    image_data = data.get('image')
    
    

    
         
   
    img2 = base64_to_image(image_data).convert('RGB')
    
    img2.save('./temp/temp.jpg')
    # parameters = {"width": "60", "height": "60", "width_asy": "0", "height_asy": "0", "tag": "", "folder_option": False, "single_face_option": True}
    face_crop = FaceCrop(width= 300, height= 250, width_asy= 0, height_asy= 0, tag= "")
    # face_crop.__init__(**parameters)
    face_crop.crop_save('./temp/temp.jpg','./temp', './tempout')
    image =  image_to_base64('./tempout/temp/temp.jpg')
    # print(image)
   
    # imgBytes=  image.tobytes()
    # buffered = io.BytesIO(imgBytes)
    # image.save(buffered, format="JPEG")
    # # print(buffered)
    # img_bytes = buffered.getvalue()

    payload = { "image": image}   

    return jsonify(payload)

@application.route('/getSentence', methods=['POST'])
def get_sentence():
    data = request.json

    name = data.get('name')
    theme = data.get('theme')
    transcribed_sentence = data.get('transcribed_sentence')
    language = data.get('language')
    

    if (theme == 'drinking'):
        theme = 'drinking water'
    db = {}
    # f = open('./data/phraseDataBase.json')
   
    # returns JSON object as 
    # a dictionary
    ppldb = {}
   
    # file_path = os.path.join('.', 'data', 'phraseDataBase.json')
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     db = json.load(file)
    
    
   
    # random_number = random.randint(0, len(db['en'][theme]['yes']))
    # response =  db['en'][theme]['yes'][random_number]
    text = str('Generate a short simple sentence as if a friend is speaking to a person who is ' + theme + ', and output only the sentence itself')
    
    if theme == 'facerecognition':
        text = 'Generate a short simple sentence that greets a person, and output only the sentence itself.'
    if (transcribed_sentence is not None and transcribed_sentence != ""):
        text = 'Generate a short simple sentence that would be a friendly answer to "'+ transcribed_sentence +'" and output only the sentence itself. '   
        
    print(text)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text}
    ]
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = llm_tokenizer(text, return_tensors="pt").to('cuda')

    generated_ids = llm_model.generate(
        **model_inputs,
        max_new_tokens=100,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    generated_ids = None
    print(response)
    
    if name != "":
        response = response + ', ' + name
    translation_tokenizer.src_lang = "en_XX"
    encoded_hi = translation_tokenizer(str(response), return_tensors="pt").to('cuda')

    if (language == 'ru'):
        generated_tokens = translation_model.generate(
            **encoded_hi,
            max_new_tokens=100,
            forced_bos_token_id=translation_tokenizer.lang_code_to_id["ru_RU"]
        )
        text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        generated_tokens = None
    elif (language == 'he'):
        generated_tokens = translation_model.generate(
            **encoded_hi,
            max_new_tokens=100,
            forced_bos_token_id=translation_tokenizer.lang_code_to_id["he_IL"]
        )
        text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        generated_tokens = None

    elif (language == 'en'):
        generated_tokens = None
        text = response
    
    torch.cuda.empty_cache()
    return jsonify({"answer": text})
# Set pad_token_id

@application.route('/voice', methods=['POST'])
def voice():
    data = request.json
    language = data.get("language")
    text = data.get("answer")

    if (language == 'ru'):
        print(str(text))
        inputs = voice_tokenizer_ru(text=str(text), return_tensors="pt").to('cuda')
        with torch.no_grad():
            
            outputs = voice_model_ru(**inputs).waveform
            audio_array = outputs.cpu().numpy().squeeze()
            encodedAudio = encode_audio_array_to_base64(audio_array ,rate=voice_model_ru.config.sampling_rate)
            torch.cuda.empty_cache()
            return jsonify({"audio": encodedAudio})
    if (language == 'he'):
        inputs = voice_tokenizer_he(text=str(text), return_tensors="pt").to('cuda')
        with torch.no_grad():

            outputs = voice_model_he(**inputs).waveform
            audio_array = outputs.cpu().numpy().squeeze()
            encodedAudio = encode_audio_array_to_base64(audio_array ,rate=voice_model_he.config.sampling_rate)
            torch.cuda.empty_cache()
            return jsonify({"audio": encodedAudio})
    else:
        inputs = voice_tokenizer_en(text=str(text), return_tensors="pt", ).to('cuda')
        with torch.no_grad():

            outputs = voice_model_en(**inputs).waveform
            audio_array = outputs.cpu().numpy().squeeze()
            encodedAudio = encode_audio_array_to_base64(audio_array ,rate=voice_model_en.config.sampling_rate)
            torch.cuda.empty_cache()
            return jsonify({"audio": encodedAudio})
        
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
    inputs = image_clsf_processor(image, str(text), return_tensors="pt").to('cuda')
    out = image_clsf_model.generate(**inputs)
    # image.show()  
    answer  = image_clsf_processor.decode(out[0], skip_special_tokens=True)
    print(answer)
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
    
    
    question =  db['en']['facerecognition']['question']
    print(str(name))
    second_image_data =  image_to_base64(str('./data/images/'+str(name).lower()+'.jpg'))
    combined = combine_images(image_data,second_image_data )
    try:
        
        
        aurus = base64.b64decode(combined)
        pred = io.BytesIO(aurus)
        image = Image.open(pred) 

    except Exception as e:
        application.logger.error(f"Error decoding image: {e}")
        return jsonify({'error': str(e)}), 400
    print(question)

    inputs = image_clsf_processor(image, question, return_tensors="pt", padding=True).to('cuda')
    out = image_clsf_model.generate(**inputs)
    # image.show()  
    answer  = image_clsf_processor.decode(out[0], skip_special_tokens=True)
    print(answer)
    return jsonify({'answer': answer })

@application.route('/analyzeMoment', methods=['POST'])
def analyze_moment():
    data = request.json
    
    image_data = data.get('image')

    db={}
    
    file_path = os.path.join('.', 'data', 'phraseDataBase.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        db = json.load(file)
    themes = []
    for key in db['en'].keys(): 
        themes.append(key)
        
    themes.remove('facerecognition')

    
    text = 'what action is the person doing on the photo?'

    # image_to_base64('./images/')
    try:
        
        
        aurus = base64.b64decode(image_data)
        pred = io.BytesIO(aurus)
        image = Image.open(pred) 

    except Exception as e:
        application.logger.error(f"Error decoding image: {e}")
        return jsonify({'error': str(e)}), 400
    print(str(text))
    clsf_inputs = image_clsf_processor(image, str(text), return_tensors="pt").to('cuda')
    clsf_out = image_clsf_model.generate(**clsf_inputs,  max_new_tokens=512)

    # image.show()  
    answer  = image_clsf_processor.decode(clsf_out[0], skip_special_tokens=True)
    print(answer)
    torch.cuda.empty_cache()
    str(themes).__contains__(answer)
    if str(themes).__contains__(answer):
        
        return jsonify({'answer': answer })
    return jsonify({'answer': 'no' })

@application.route('/recognize', methods=['POST'])
def recognize():
 

    files = {"file": open('C:/Users/hamme/serv/server/tempout/temp/temp.jpg', 'rb')}
    recres = requests.post('http://localhost:8050/api/v1/recognition/recognize?limit=5&prediction_count=1', files=files,headers={"x-api-key": '2df50b47-0d98-4363-9899-cf683f184de0'})
    # inputs = image_clsf_processor(image, question, return_tensors="pt", padding=True).to('cuda')
    # out = image_clsf_model.generate(**inputs)
    name = 'no' 
    results = recres.json()
    if results is not None and results.get("result") is not None and len(results["result"]) > 0:
        # print(recres.result[0].box.subjects[0].subject)
        name = results["result"][0]["subjects"][0]["subject"]
        print(name)
    return jsonify({'answer': name })

@application.route('/speech', methods=['POST'])
def speech():
   
    # Open file and write binary (blob) data
    f = open('./file.wav', 'wb')
    f.write(request.data)
    f.close()
    
    loaded = whisper.load_audio("C:/Users/hamme/serv/server/file.wav")
    # audio = whisper.pad_or_trim(audio)
    result = speech_recog_model.transcribe(loaded, verbose=True)
    if check_key_in_json(result, "segments"):
        return jsonify({'answer': result["segments"][-1]["text"] })
    
    return jsonify({'answer': 'no' })
 
@application.route('/depth', methods=['POST'])
def depthEstimation():
    data = request.json
    image_data = data.get('image')
    try:
        
        
        aurus = base64.b64decode(image_data)
        pred = io.BytesIO(aurus)
        image = Image.open(pred) 

    except Exception as e:
        application.logger.error(f"Error decoding image: {e}")
        return jsonify({'error': str(e)}), 400
    inputs = image_processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.show()

    depth.save("depth_map.png")
    depth_image = image_to_base64('depth_map.png')
    return jsonify({'answer': depth_image })  
    