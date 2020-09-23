import numpy as np 
import random



def generate_real_samples(dataset,num_samples,patch_shape):
	train_a,train_b=dataset

	random_index=np.random.randint(0,train_a.shape[0],num_samples)

	img1,img2=train_a[random_index],train_b[random_index]

	y=np.ones((num_samples,patch_shape,patch_shape,1))

	return [img1,img2],y


def generate_fake_samples(generator_model,samples,patch_shape):

	fake=generator_model.predict(samples)

	y=np.zeros((len(fake),patch_shape,patch_shape,1))

	return fake,y


