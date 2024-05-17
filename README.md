# Neural-Style-Transfer
 

1. Model Used: 
The code is using the "Arbitrary Image Stylization" model provided by Google Magenta through TensorFlow Hub. This model allows for arbitrary style transfer, meaning you can apply the style of one image onto another image regardless of their content.


2. This model is trained on two datasets:
  1. Imagenet (https://www.kaggle.com/c/imagenet-object-localization-challenge/overview)
  2. COCONet (Common Obejct Context)   (https://cocodataset.org)


Based on the model code in [magenta](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization) and the publication:

[Exploring the structure of a real-time, arbitrary neural artistic stylization
network](https://arxiv.org/abs/1705.06830).
*Golnaz Ghiasi, Honglak Lee,
Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens*,
Proceedings of the British Machine Vision Conference (BMVC), 2017.


### 🎨🖌 Creating Art with the help of Artificial Intelligence !

**🔥 Official Website :** https://nst-one-infix.streamlit.app/


</br>

Neural Style Transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation. Popular use cases for NST are the creation of artificial artwork from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs.

<br> <!-- line break -->

<div align="center">
<img src="/Imgs/nst.png"/>
</div>

<br> <!-- line break -->


## 📝 Summary of Neural Style Transfer

Style transfer is a computer vision technique that takes two images — a "content image" and "style image" — and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image. Training a style transfer model requires two networks,which follow a encoder-decoder architecture : 
- A pre-trained feature extractor 
- A transfer network


<div align="center">
<img src="/Imgs/nst architecture.jpg" width="80%"/>
</div>

<br> <!-- line break -->



The ‘encoding nature’ of CNN’s is the key in Neural Style Transfer. Firstly, we initialize a noisy image, which is going to be our output image(G). We then calculate how similar is this image to the content and style image at a particular layer in the network(VGG network). Since we want that our output image(G) should have the content of the content image(C) and style of style image(S) we calculate the loss of generated image(G) w.r.t to the respective content(C) and style(S) image.



<div align="center">
<img src="/Imgs/final_oss.png" width="50%" />
</div>

<br> <!-- line break -->


In simple words,we optimize our NST models to reduce the 'content loss' and the 'style loss'. The content loss function ensures that the activations of the higher layers are similar between the content image and the generated image. The style loss function makes sure that the correlation of activations in all the layers are similar between the style image and the generated image.


## 👨‍💻 Implementation

Early versions of NST treated the task as an optimization problem, requiring hundreds or thousands of iterations to perform style transfer on a single image. To tackle this inefficiency, researchers developed what’s referred to as "Fast Neural Style Transfer". Fast style transfer also uses deep neural networks but trains a standalone model to transform any image in a single, feed-forward pass. Trained models can stylize any image with just one iteration through the network, rather than thousands.State-of-the-art style transfer models can even learn to imprint multiple styles via the same model so that a single input content image can be edited in any number of creative ways.

In this project we used a pre-trained "Arbitrary Neural Artistic Stylization Network" - a Fast-NST architecture which you can find [here](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2). The model is successfully trained on a corpus of roughly 80,000 paintings and is able to generalize to paintings previously unobserved.




<div align="center">
  <img src="/Imgs/content1.jpg" width="35%"/>
  <img src="/Imgs/art1.png" width="35%"/>
</div>

<div align="center">
  <img src="/Imgs/content2.jpg" width="35%"/>
  <img src="/Imgs/art2.png" width="35%"/>
</div>

<div align="center">
  <img src="/Imgs/content3.jpg" width="35%"/>
  <img src="/Imgs/art3.png" width="35%"/>
</div>

<div align="center">
  <img src="/Imgs/content4.jpg" width="35%"/>
  <img src="/Imgs/art4.png" width="35%"/>
</div>


References :
- https://arxiv.org/abs/1508.06576 
- https://keras.io/examples/generative/neural_style_transfer/ 
- https://arxiv.org/abs/1705.06830 
- https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2 















