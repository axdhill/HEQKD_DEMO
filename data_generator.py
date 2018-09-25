import numpy as np
import time

block_size = 100000
density = 1000
resolution = 156e-12
frame = 0
f = open('data_input.txt', 'w')

while(True) :
	ttags = resolution*np.random.uniform( (block_size*frame), (block_size*(frame+1)), density )
	ttags = np.sort(ttags)

	chs = np.random.choice([0,1,2,3], len(ttags))
	print(np.repeat([frame], len(ttags)))
	print(ttags)
	write_array = np.vstack((np.repeat([frame], len(ttags)), chs, ttags))
	print(write_array.T)

	frame += 1
	np.savetxt(f, write_array.T)
	time.sleep(0.5)

f.close()


