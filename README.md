# Image-style-transfer
 This web app will help us to implement the image style transfer from one image to another.
 

1. Model Used: 
The code is using the "Arbitrary Image Stylization" model provided by Google Magenta through TensorFlow Hub. This model allows for arbitrary style transfer, meaning you can apply the style of one image onto another image regardless of their content.

2. This model is trained on two datasets:
  1. Imagenet
  2. COCO (Common Obejct COntext)

Based on the model code in [magenta](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization) and the publication:

[Exploring the structure of a real-time, arbitrary neural artistic stylization
network](https://arxiv.org/abs/1705.06830).
*Golnaz Ghiasi, Honglak Lee,
Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens*,
Proceedings of the British Machine Vision Conference (BMVC), 2017.