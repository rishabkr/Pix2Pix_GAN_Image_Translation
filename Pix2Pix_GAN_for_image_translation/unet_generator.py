import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from tensorflow.keras.initializers import RandomNormal
from keras.models import Model,Input
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,LeakyReLU,Activation,Concatenate
from tensorflow.keras.layers import Dropout,BatchNormalization
from keras.utils.vis_utils import plot_model

def encoder_block(input_layer,num_filters,batch_normalization=True):
	init=RandomNormal(stddev=0.02)
	generator=Conv2D(num_filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(input_layer)
	if(batch_normalization):
		generator=BatchNormalization()(generator,training=True)

	generator=LeakyReLU(alpha=0.2)(generator)
	return generator



def deccoder_block(input_layer,skip_input,num_filters,dropout=True):
	init=RandomNormal(stddev=0.02)
	generator=Conv2DTranspose(num_filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(input_layer)
	generator=BatchNormalization()(generator,training=True)
	if(dropout):
		generator=Dropout(0.5)(generator,training=True)

	generator=Concatenate()([generator,skip_input])

	generator=Activation('relu')(generator)
	return generator




def build_generator(image_shape=(256,256,3)):
	init=RandomNormal(stddev=0.02)
	input_image=Input(shape=image_shape)
	filters=[64,128,256,512]
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	encoder_layer1=encoder_block(input_image,filters[0],batch_normalization=False)

	encoder_layer2=encoder_block(encoder_layer1,filters[1])
	encoder_layer3=encoder_block(encoder_layer2,filters[2])
	encoder_layer4=encoder_block(encoder_layer3,filters[3])

	encoder_layer5=encoder_block(encoder_layer4,filters[3])
	encoder_layer6=encoder_block(encoder_layer5,filters[3])
	encoder_layer7=encoder_block(encoder_layer6,filters[3])

	bottleneck_layer=Conv2D(filters[3],(4,4),strides=(2,2),padding='same',kernel_initializer=init)(encoder_layer7)
	bottleneck_layer=Activation('relu')(bottleneck_layer)


	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

	decoder_layer1=deccoder_block(bottleneck_layer,encoder_layer7,filters[3])
	decoder_layer2=deccoder_block(decoder_layer1,encoder_layer6,filters[3])
	decoder_layer3=deccoder_block(decoder_layer2,encoder_layer5,filters[3])
	
	decoder_layer4=deccoder_block(decoder_layer3,encoder_layer4,filters[3],dropout=False)
	decoder_layer5=deccoder_block(decoder_layer4,encoder_layer3,filters[2],dropout=False)
	decoder_layer6=deccoder_block(decoder_layer5,encoder_layer2,filters[1],dropout=False)
	decoder_layer7=deccoder_block(decoder_layer6,encoder_layer1,filters[0],dropout=False)
	
	generator=Conv2DTranspose(3,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(decoder_layer7)

	output_image=Activation('tanh')(generator)

	model=Model(input_image,output_image)
	return model


if __name__=='__main__':
	image_shape=(256,256,3)
	model=build_generator(image_shape)

	model.summary()
	plot_model(model,to_file='UNet_Model.png',show_shapes=True,show_layer_names=True)


