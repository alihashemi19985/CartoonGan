# CartoonGan
"CartoonGAN" refers to a type of Generative Adversarial Network (GAN) designed specifically for generating cartoon-style images. GANs are a class of deep learning models consisting of two neural networks, a generator and a discriminator, which are trained simultaneously in a competitive manner. 
The goal of CartoonGAN is to learn the mapping from real-world images to cartoon-style images. It takes a real image as input and generates a corresponding cartoon-style image as output. The network is trained on a dataset containing pairs of real images and their corresponding cartoon-style images. During training, the generator learns to produce realistic cartoon-style images that are indistinguishable from the real ones, while the discriminator learns to distinguish between real and generated images.

Pytorch
First, you must download the pre-trained models with the command below, you will download these models. copy in the terminal:
sh pretrained_model/download_pth.sh
Then run the program, and copy in the terminal: 
streamlit run st-app.py 
