import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tqdm import tqdm
import matplotlib.pyplot as plt 

def load_images(path,size=(256,512)):
	source_list=[]
	target_list=[]
	for filename in tqdm(os.listdir(path)):
		image=load_img(path+filename,target_size=size)
		image=img_to_array(image)

		sattelite_image,map_image=image[:,:256],image[:,256:]

		source_list.append(sattelite_image)
		target_list.append(map_image)

	return [np.asarray(source_list),np.asarray(target_list)]

def load_real_samples(filename):
	dataset=np.load(filename)
	src_image,tar_images=dataset['arr_0'],dataset['arr_1']
	src_image=(src_image-127.5)/127.5
	tar_images=(tar_images-127.5)/127.5

	return src_image,tar_images



if __name__=='__main__':
	path=r'D:/datasets/Maps/maps/train/'
	
	[source_images,target_images]=load_images(path)

	print(f'source_images shape {source_images.shape}  target_images shape {target_images.shape}')

	file_name='maps_splitted_256.npz'
	np.savez_compressed(file_name,source_images,target_images)
	print(f'File saved as {file_name}')
	print('-'*50)

	dataset=np.load(file_name)

	src_image,tar_images=dataset['arr_0'],dataset['arr_1']

	num_samples=5

	for i in range(num_samples):
		plt.subplot(2,num_samples,1+i)
		plt.axis('off')
		plt.imshow(src_image[i].astype('uint8'))

	for i in range(num_samples):
		plt.subplot(2,num_samples,1+num_samples+i)
		plt.axis('off')
		plt.imshow(tar_images[i].astype('uint8'))

	plt.show()
