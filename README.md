# Pix2Pix_GAN_Image_Translation
An implemenation of Pix2Pix GAN for google satellite images to stree view translation
U-Net Generator Model
•	The generator model takes an image as input, and unlike a standard GAN model, it does not take a point from the latent space as input. Instead, the source of randomness comes from the use of dropout layers that are used both during training and when a prediction is made.
•	The U-Net model architecture is very similar in that it involves downsampling to a bottleneck and upsampling again to an output image, but links or skip-connections are made between layers of the same size in the encoder and the decoder, allowing the bottleneck to be circumvented.

PatchGAN Discriminator Model
•	The discriminator model takes an image from the source domain and an image from the target domain and predicts the likelihood of whether the image from the target domain is a real or generated version of the source image.
•	The input to the discriminator model highlights the need to have an image dataset comprised of paired source and target images when training the model. Unlike the standard GAN model that uses a deep convolutional neural network to classify images, the Pix2Pix model uses a PatchGAN. This is a deep convolutional neural network designed to classify patches of an input image as real or fake, rather than the entire image.
•	A patch size of 70 × 70 was found to be effective across a range of image-to-image translation tasks.

Composite Adversarial and L1 Loss
•	The training of the discriminator is too fast compared to the generator, therefore the discriminator loss is halved in order to slow down the training process.
•	 Discriminator Loss = 0.5 × Discriminator Loss
•	The generator model is trained using both the adversarial loss for the discriminator model and the L1 or mean absolute pixel difference between the generated translation of the source image and the expected target image. The adversarial loss and the L1 loss are combined into a composite loss function, which is used to update the generator model. L2 loss was also evaluated and found to result in blurry images.
•	The adversarial loss influences whether the generator model can output images that are plausible in the target domain, whereas the L1 loss regularizes the generator model to output images that are a plausible translation of the source image.
•	Generator Loss = Adversarial Loss + λ × L1 Loss


The PatchGAN is designed based on the size of the receptive field, sometimes called the effective receptive field. The receptive field is the relationship between one output activation of the model to an area on the input image (actually volume as it proceeded down the input channels). A PatchGAN with the size 70 × 70 is used, which means that the output (or each output) of the model maps to a 70 × 70 square of the input image. In effect, a 70 × 70 PatchGAN will classify 70 × 70 patches of the input image as real or fake.

The receptive field is not the size of the output of the discriminator model, e.g. it does not refer to the shape of the activation map output by the model. It is a definition of the model in terms of one pixel in the output activation map to the input image. The output of the model may be a single value or a square activation map of values that predict whether each patch of the input image is real or fake.
The calculation of the receptive field in one dimension is calculated as: 
receptive field = (output size − 1) × stride + kernel size 
Where output size is the size of the prior layers activation map, stride is the number of pixels the filter is moved when applied to the activation, and kernel size is the size of the filter to be applied. The PatchGAN uses a fixed stride of 2 × 2 (except in the output and second last layers) and a fixed kernel size of 4 × 4.
The PatchGAN configuration is defined using a shorthand notation as: C64-C128-C256- C512, where C refers to a block of Convolution-BatchNorm-LeakyReLU layers and the number indicates the number of filters. Batch normalization is not used in the first layer.

In UNet Generator model, Skip connections are added between the layers with the same sized feature maps so that the first downsampling layer is connected with the last upsampling layer, the second downsampling layer is connected with the second last upsampling layer, and so on.
batch normalization is used in the same way during training and inference, meaning that statistics are calculated for each batch and not fixed at the end of the training process. This is referred to as instance normalization, specifically when the batch size is set to 1 as it is with the Pix2Pix model.
 The encoder uses blocks of Convolution-BatchNorm-LeakyReLU like the discriminator model, whereas the decoder model uses blocks of Convolution-BatchNorm-Dropout-ReLU with a dropout rate of 50%. All convolutional layers use a filter size of 4 × 4 and a stride of 2 × 2.

The architecture of the U-Net model is defined using the shorthand notation as:
  Encoder: C64-C128-C256-C512-C512-C512-C512-C512
  Decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

The discriminator model can be updated directly, whereas the generator model must be updated via the discriminator model. The discriminator model can then predict whether a generated image is real or fake. n update the weights of the composite model in such a way that the generated image has the label of real instead of fake, which will cause the generator weights to be updated towards generating a better fake image. We can also mark the discriminator weights as not trainable in this context, to avoid the misleading update.



