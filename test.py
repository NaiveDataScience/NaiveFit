import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


## The architecture of the network
class DoubleReluNet(nn.Module):

    def __init__(self):
        super(DoubleReluNet, self).__init__()
        
        self.fc1 = nn.Linear(1, 20)  ## Input layer
        # self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(20, 1) ## Output layer

    def forward(self, x):
        
        x = F.relu(self.fc1(x))		## Relu layer
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)			
        return x


##Sampling data with different function
def get_square_data(batch_size=100):
	x = np.random.randn(batch_size, 1).astype(np.float64)
	x.sort(axis=0) 
	return x * 10

def get_sin_data(batch_size=100):
	x = np.random.random(batch_size).astype(np.float64).reshape(batch_size, 1)
	x.sort(axis=0) 
	return x * 6

def get_expo_data(batch_size=100):
	x = np.random.random(batch_size).astype(np.float64).reshape(batch_size, 1)
	x.sort(axis=0) 
	return x * 6

def get_ln_data(batch_size=100):
	x = np.random.random(batch_size).astype(np.float64).reshape(batch_size, 1)
	x.sort(axis=0) 
	return x * 50

def get_sin_plus_cos_data(batch_size=100):
	x = np.random.random(batch_size).astype(np.float64).reshape(batch_size, 1)
	x.sort(axis=0) 
	return x * 6


##Get the target tensor from function and input
def get_batch(f, inp):

	expect = np.vectorize(f)(inp)
	print(expect)
	return torch.from_numpy(inp), torch.from_numpy(expect)


##Training(Fitting) the data
def fit(func, Net, inp, expect, train_config, save_name=''):
	net = Net().double()
	optimizer = optim.Adam(net.parameters(), 
					lr = train_config["learning_rate"])  
	
	## Training Process
	for i in range(train_config["epoch_number"]):
		optimizer.zero_grad()   # zero the gradient buffers
		output = net(inp)

		loss = nn.MSELoss()(output, expect)
		loss.backward()
		optimizer.step()    # Does the update
		if i % 10000 == 0:
			print (loss)

	if save_name:	##Whether save the current picutre
		torch.save(net, save_name)
	return net


##General function to draw the fitting curve picture
def test(net, test_data, f, plt_config):
	numpy_y = net(test_data).detach().numpy().T[0]
	numpy_x = test_data.detach().numpy().T[0]

	plt.figure(figsize=(12, 8))
	plt.plot(numpy_x, numpy_y, color='red', label='fitting', lw='2')
	plt.plot(numpy_x, list(map(f, numpy_x)), color='blue', label='expecting', lw='2')

	plt.legend()
	plt.title(plt_config["pic_title"])
	plt.savefig(plt_config["pic_name"])



##The followings are all the function that will be used.
def square(x):
	return x * x

def sino(x):
	return math.sin(x)

def expo(x):
	return math.exp(x)

def ln(x):
	return math.log(x)

def sin_plus_cos(x):
	return math.sin(x) + math.cos(x)


##Merge all the pictures into one
def plot_all():
	plt.figure()
	HEIGHT = 800
	WIDTH = 1200

	filenames = [
		"y = expo: Train.png",
		"y = expo: Test.png",
		"y = ln x: Train.png",
		"y = ln x: Test.png",
		"y = sin x: Train.png",
		"y = sin x: Test.png",
		"y = sinx + cosx: Train.png",
		"y = sinx + cosx: Test.png",
		"y = x ^ 2: Train.png",
		"y = x ^ 2: Test.png"
	]

	target = Image.new('RGB', (WIDTH*2, HEIGHT*5))
	for (i, name) in enumerate(filenames):
		target.paste(Image.open(name),
			(WIDTH * (i % 2), HEIGHT * (i//2)))
	
	target.save("merge.png")



if __name__ == '__main__':


	
	sample = get_square_data() #Get sampling data from function
	inp, expect = get_batch(square, sample)

	##Fit the data
	net = fit(square, DoubleReluNet, inp, expect, {
		"learning_rate": 0.01,
		"epoch_number": 100000 
		}, save_name="square.pkl")

	##Plot training data
	test(net, inp, square, {
		"pic_name" : "y = x ^ 2: Train",
		"pic_title": "y = x ^ 2: Train"
		})

	##Sample testing data
	test_data = get_square_data()
	net = torch.load("square.pkl")
	##Plot testing data
	test(net, test_data, square, {
		"pic_name" : "y = x ^ 2: Test",
		"pic_title": "y = x ^ 2: Test"
		})




	sample = get_expo_data(batch_size=1000)
	inp, expect = get_batch(expo, sample)
	net = fit(expo, DoubleReluNet, inp, expect, {
		"learning_rate": 0.00001,
		"epoch_number": 500000 
		}, save_name="expo.pkl")

	test(net, inp, expo, {
		"pic_name" : "y = expo: Train",
		"pic_title": "y = expo: Train"
		})

	test_data = get_sin_data()
	inp, expect = get_batch(expo, test_data)
	net = torch.load("expo.pkl")

	test(net, inp, expo, {
		"pic_name" : "y = expo: Test",
		"pic_title": "y = expo: Test"
		})


	sample = get_sin_data(batch_size=1000)
	inp, expect = get_batch(sino, sample)
	net = fit(sino, DoubleReluNet, inp, expect, {
		"learning_rate": 0.00001,
		"epoch_number": 100000 
		}, save_name="sinx.pkl")

	test(net, inp, sino, {
		"pic_name" : "y = sin x: Train",
		"pic_title": "y = sin x: Train"
		})

	test_data = get_sin_data()
	inp, expect = get_batch(sino, test_data)
	net = torch.load("sinx.pkl")

	test(net, inp, sino, {
		"pic_name" : "y = sin x: Test",
		"pic_title": "y = sin x: Test"
		})

	sample = get_ln_data(batch_size=1000)
	inp, expect = get_batch(ln, sample)
	net = fit(ln, DoubleReluNet, inp, expect, {
		"learning_rate": 0.00001,
		"epoch_number": 100000 
		}, save_name="ln.pkl")

	test(net, inp, ln, {
		"pic_name" : "y = ln x: Train",
		"pic_title": "y = ln x: Train"
		})

	test_data = get_ln_data()
	inp, expect = get_batch(ln, test_data)
	net = torch.load("ln.pkl")

	test(net, inp, ln, {
		"pic_name" : "y = ln x: Test",
		"pic_title": "y = ln x: Test"
		})

	sample = get_sin_plus_cos_data(batch_size=1000)
	inp, expect = get_batch(sin_plus_cos, sample)
	net = fit(sin_plus_cos, DoubleReluNet, inp, expect, {
		"learning_rate": 0.00001,
		"epoch_number": 100000 
		}, save_name="sinx_plus_cosx.pkl")

	test(net, inp, sin_plus_cos, {
		"pic_name" : "y = sinx + cosx: Train",
		"pic_title": "y = sinx + cosx: Train"
		})

	test_data = get_sin_plus_cos_data()
	inp, expect = get_batch(sin_plus_cos, test_data)
	net = torch.load("sinx_plus_cosx.pkl")

	test(net, inp, sin_plus_cos, {
		"pic_name" : "y = sinx + cosx: Test",
		"pic_title": "y = sinx + cosx: Test"
		})
	plot_all()



