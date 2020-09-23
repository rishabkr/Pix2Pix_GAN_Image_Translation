import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from patch_gan import build_discriminator
from unet_generator import build_generator
from receptive_field import receptive_field
from keras.utils.vis_utils import plot_model
# from tensorflow.keras.utils import plot_model



def build_pix2pix_gan(generator_model,discriminator_model,image_shape):
	discriminator_model.trainable=False

	input_source_image=Input(shape=image_shape)

	generator_output=generator_model(input_source_image)

	discriminator_output=discriminator_model([input_source_image,generator_output])

	model=Model(input_source_image,[discriminator_output,generator_output])

	optimizer=Adam(lr=0.0002,beta_1=0.5,beta_2=0.999)

	#loss = adversarial loss + λ × L1 loss
	model.compile(loss=['binary_crossentropy','mae'],optimizer=optimizer,loss_weights=[1,100])

	return model


if __name__=='__main__':
	image_shape=(256,256,3)
	
	discriminator_model=build_discriminator(image_shape)

	generator_model=build_generator(image_shape)

	gan_model=build_pix2pix_gan(generator_model,discriminator_model,image_shape)

	gan_model.summary()

	# plot_model(gan_model,to_file='pix2pix_gan.png',show_shapes=True,show_layer_names=True)
