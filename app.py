from flask import Flask, request, jsonify, render_template
from transformers import VisionEncoderDecoderModel
from transformers import ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
tokenize = AutoTokenizer.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-caption', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})
    file = request.files['file']
    num_return_sequences = int(request.form.get('num_return_sequences'))
    img = Image.open(file)
    all_captions = []
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')
    image_captions = []
    gen_kwargs = {"max_length": 60, "num_beams": num_return_sequences +
                  2, "num_return_sequences": num_return_sequences}

    output_ids = model.generate(
        pixel_values=feature_extractor(
            images=img, return_tensors='pt').pixel_values.to(device),
        **gen_kwargs
    )
    for output_id in output_ids:

        caption = tokenize.decode(output_id, skip_special_tokens=True).strip()
        image_captions.append(caption)
    all_captions.append(image_captions)
    return jsonify({'captions': all_captions})


if __name__ == '__main__':
    app.run(debug=True)
