def receptive_field(output_size,kernel_size,stride_size):
	return (output_size-1)*stride_size+kernel_size

if __name__=='__main__':
	for i in range(4):
		print('-'*25)
		print(f'Enter value of output_size, kernel_size and stride_size for demonstration {i} :')
		output_size=int(input())
		kernel_size=int(input())
		stride_size=int(input())
		print('-'*25)
		print(receptive_field(output_size,kernel_size,stride_size))