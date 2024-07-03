from flask import Flask, request, jsonify
from transformers import AutoProcessor, BarkModel
import scipy
from werkzeug.middleware.proxy_fix import ProxyFix
application = Flask(__name__)

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to("cuda")
model.eval()


application.wsgi_app = ProxyFix(
    application.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)
@application.route('/voice', methods=['POST'])
def voice():
    data = request.json
    text = data.get('text')
    language = data.get('language')
    voice_preset = "v2/ru_speaker_1"

    inputs = processor("Неплохай", voice_preset=voice_preset).to("cuda")    
    # inputs = processor2(text, voice_preset=voice_preset)

    # audio_array = model2.generate(**inputs)
    # processor2.decode(audio_array[0], skip_special_tokens=True)
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    # resp = requests.post("http://localhost:8888/queue/join?", json=payload)
    # audioUrl = get_stream("http://localhost:8888/queue/data?session_hash=dca35llyreq")
    # audioblob = requests.get(audioUrl)
    # audio_base64 = encode_audio_from_url_to_base64(audioUrl)
    
    return jsonify({"audio":audio_array})

