from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from keras.models import Model,Input
from tensorflow.keras.layers import Conv2D,LeakyReLU,Activation,Concatenate,BatchNormalization
from keras.utils.vis_utils import plot_model

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def build_discriminator(image_shape):
	weight_init=RandomNormal(stddev=0.02)

	input_source_image=Input(shape=image_shape)

	input_target_image=Input(shape=image_shape)

	#concatenate
	merge=Concatenate()([input_source_image,input_target_image])
	#C64 #Layer 1 doesnt have BN layer
	disc=Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=weight_init)(merge)
	disc=LeakyReLU(alpha=0.2)(disc)

	#C128
	disc=Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer=weight_init)(disc)
	disc=BatchNormalization()(disc)
	disc=LeakyReLU(alpha=0.2)(disc)
	#C256
	disc=Conv2D(256,(4,4),strides=(2,2),padding='same',kernel_initializer=weight_init)(disc)
	disc=BatchNormalization()(disc)
	disc=LeakyReLU(alpha=0.2)(disc)
	#C512
	disc=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=weight_init)(disc)
	disc=BatchNormalization()(disc)
	disc=LeakyReLU(alpha=0.2)(disc)


	#second last output layer
	disc=Conv2D(512,(4,4),padding='same',kernel_initializer=weight_init)(disc)
	disc=BatchNormalization()(disc)
	disc=LeakyReLU(alpha=0.2)(disc)

	#outputlayer
	disc=Conv2D(1,(4,4),padding='same',kernel_initializer=weight_init)(disc)
	patch_output=Activation('sigmoid')(disc)

	model=Model([input_source_image,input_target_image],patch_output)
	#adam according to paper
	optimizer=Adam(lr=0.0002,beta_1=0.5,beta_2=0.999)

	model.compile(loss='binary_crossentropy',optimizer=optimizer,loss_weights=[0.5])
	return model

if __name__=='__main__':
	image_shape=(256,256,3)

	model=build_discriminator(image_shape)

	model.summary()
	plot_model(model,to_file='patch_gan_discriminator.png',show_shapes=True,show_layer_names=True)

