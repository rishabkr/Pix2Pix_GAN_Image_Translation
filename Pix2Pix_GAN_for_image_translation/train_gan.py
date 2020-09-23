import numpy as np 
import tensorflow as tf
from sample_generator import generate_real_samples,generate_fake_samples
from tensorflow.keras.initializers import RandomNormal
from keras.models import Model,Input
from build_gan import build_pix2pix_gan
from patch_gan import build_discriminator
from unet_generator import build_generator
from sample_loader import load_images,load_real_samples
from tqdm import tqdm
import matplotlib.pyplot as plt

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

def train_gan(discriminator_model,generator_model,gan_model,dataset,num_epochs=100,num_batch=1,num_patch=16):
	train_a,train_b=dataset

	batch_per_epochs=int(len(train_a)/num_batch)

	num_steps=batch_per_epochs*num_epochs

	for i in tqdm(range(num_steps)):
		#generate real samples
		[x_real_a,x_real_b],y_real=generate_real_samples(dataset,num_batch,num_patch)

		#generate fake samples
		x_fake_b,y_fake=generate_fake_samples(generator_model,x_real_a,num_patch)

		 #update disccriminator

		discriminator_loss_1=discriminator_model.train_on_batch([x_real_a,x_real_b],y_real)

		discriminator_loss_2=discriminator_model.train_on_batch([x_real_a,x_real_b],y_fake)

		generator_loss,_,_=gan_model.train_on_batch(x_real_a,[y_real,x_real_b])

		print(f'step {i+1} discriminator_loss_1 : {discriminator_loss_1}  discriminator_loss_2 : {discriminator_loss_2}  generator_loss : {generator_loss}')

		if(i+1)%(batch_per_epochs*10)==0:
			evaluate_and_see_performance(i,generator_model,dataset)



def evaluate_and_see_performance(step,generator_model,dataset,num_samples=5):
	
	[x_real_a,x_real_b],_=generate_real_samples(dataset,num_samples,1)

	x_fake_b,_=generate_fake_samples(generator_model,x_real_a,1)

	x_real_a=(x_real_a+1)/2.0
	x_real_b=(x_real_b+1)/2.0

	x_fake_b=(x_fake_b+1)/2.0

	for i in range(num_samples):
		plt.subplot(3, num_samples, 1 + i)
		plt.axis('off')
		plt.imshow(x_real_a[i])
		# plot generated target image
	
	for i in range(num_samples):
		plt.subplot(3, num_samples, 1 + num_samples + i)
		plt.axis('off')
		plt.imshow(x_fake_b[i])
		# plot real target image
	
	for i in range(num_samples):
		plt.subplot(3, num_samples, 1 + num_samples*2 + i)
		plt.axis('off')
		plt.imshow(x_real_b[i])

	file_1=f'models_and_results/plot_{step+1}.png'
	plt.savefig(file_1)
	plt.close()
	
	file_2=f'models_and_results/model_{step+1}.h5'
	generator_model.save(file_2)
	print(f'saved {file_1} and {file_2} ')


def successfully_built(model_name):
	print('='*75)
	print(f'successfully_built {model_name} ')
	print('='*75)

if __name__=='__main__':
	dataset_name='maps_splitted_256.npz'
	dataset=load_real_samples(dataset_name)

	print('dataset loaded ')
	print('-'*50)

	image_shape=dataset[0].shape[1:]

	discriminator_model=build_discriminator(image_shape)
	successfully_built('discriminator_model')

	generator_model=build_generator(image_shape)
	successfully_built('generator_model')

	gan_model=build_pix2pix_gan(generator_model,discriminator_model,image_shape)

	successfully_built('gan_model')

	train_gan(discriminator_model,generator_model,gan_model,dataset)

