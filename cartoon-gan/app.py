from flask import Flask, current_app, g,render_template,jsonify,request
import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer

style = 'Hayao'
model_path = './pretrained_model'
model = Transformer()
model.load_state_dict(torch.load(os.path.join(model_path, style + '_net_G_float.pth')))
model.eval()
model.float()
output_dir = 'test-app'


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload_image():


    image_file = request.files['image']
    img = Image.open(image_file)
    input_image = img.convert('RGB')
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h *1.0 / w
    
    if ratio > 1:        
        h = 450
        w = int(h*1.0/ratio)
    else:
        w = 450
        h = int(w * ratio)

    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)
	# RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = -1 + 2 * input_image
    input_image = Variable(input_image, volatile=True).float()
    output_image = model(input_image)
    output_image = output_image[0]
	# BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
	# save
    vutils.save_image(output_image, os.path.join(output_dir,   '_' + style + '.jpg'))
    

    return 'asas'







   
	

if __name__ == '__main__':
    app.run(debug=True)


