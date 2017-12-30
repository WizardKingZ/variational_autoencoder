import matplotlib.pyplot as plt
import numpy as np
"""
this set of custom plot library refers to 
http://www.cvc.uab.es/people/joans/slides_tensorflow/tensorflow_html/vae-Jan-Hendrik-Metzen.html
"""

def comparison_plot(model, sample, color=None, shape=[28, 28, 1]):
	x_reconstruct = model.recreate(sample)
	plt.figure(figsize=(8, 12))
	for i in range(5):
		plt.subplot(5, 2, 2*i + 1)
		if shape[2] == 1:
			plt.imshow(sample[i].reshape(shape[0], shape[1]), vmin=0, vmax=1, cmap=color, interpolation='nearest')
		else:
			plt.imshow(sample[i].reshape(shape[0], shape[1], shape[2]), vmin=0, vmax=1, cmap=color, interpolation='nearest')
		plt.title("Test input")
		plt.colorbar()
		plt.subplot(5, 2, 2*i + 2)
		if shape[2] == 1:
			plt.imshow(x_reconstruct[i].reshape(shape[0], shape[1]), vmin=0, vmax=1, cmap=color, interpolation='nearest')
		else:
			plt.imshow(x_reconstruct[i].reshape(shape[0], shape[1], shape[2]), vmin=0, vmax=1, cmap=color, interpolation='nearest')
		plt.title("Reconstruction")
		plt.colorbar()
	plt.tight_layout()

def latent_space_plot(model, x_sample, y_sample):
	z_mu = model.transform(x_sample)
	plt.figure(figsize=(8, 6))
	plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
	plt.colorbar()
	plt.grid()


def manifold_plot(model, sample, color=None, shape=[28, 28, 1], size=100, reconstruct=True, fig_size=(8, 8), check_point=None):
	if reconstruct:
		x_reconstruct = model.recreate(sample)
	else:
		x_reconstruct = sample
	plt.figure(figsize=fig_size)
	manifold = None
	for i in range(int(np.sqrt(size))):
		## row 
		row = None
		for j in range(int(np.sqrt(size))):
			if row is None:
				if shape[2] == 1:
					row = x_reconstruct[10*i+j].reshape(shape[0], shape[1])
				else:
					row = x_reconstruct[10*i+j].reshape(shape[0], shape[1], shape[2])
			else:
				if shape[2] == 1:
					row = np.concatenate((row, x_reconstruct[10*i+j].reshape(shape[0], shape[1])), axis=1)
				else:
					row = np.concatenate((row, x_reconstruct[10*i+j].reshape(shape[0], shape[1], shape[2])), axis=1)
		if manifold is None:
			manifold = row
		else:
			manifold = np.concatenate((manifold, row))
	plt.imshow(manifold, vmin=0, vmax=1, cmap=color, interpolation='nearest')
	plt.tight_layout()
	plt.axis('off')
	if check_point is not None:
		plt.savefig(check_point)

def two_d_manifold_plot(model, init_lower=-3, init_upper=3, size=20, color=None, shape=[28, 28, 1], fig_size=(8, 8), check_point=None):
	nx = ny = size
	x_values = np.linspace(init_lower, init_upper, nx)
	y_values = np.linspace(init_lower, init_upper, ny)

	canvas = None
	for i, yi in enumerate(x_values):
		row = None
		for j, xi in enumerate(y_values):
			z_mu = np.array([[xi, yi]]*model.mini_batch_size)
			x_mean = model.generate(z_mu)[0]
			if row is None:
				if shape[2] == 1:
					row = x_mean.reshape(shape[0], shape[1])
				else:
					row = x_mean.reshape(shape[0], shape[1], shape[2])
			else:
				if shape[2] == 1:
					row = np.concatenate((row, x_mean.reshape(shape[0], shape[1])), axis=1)
				else:
					row = np.concatenate((row, x_mean.reshape(shape[0], shape[1], shape[2])), axis=1)
		if canvas is None:
			canvas = row
		else:
			canvas = np.concatenate((canvas, row))
	plt.figure(figsize=(8, 10))        
	plt.imshow(canvas, vmin=0, vmax=1, cmap=color, interpolation='nearest')
	plt.tight_layout()
	plt.axis('off')
	if check_point is not None:
		plt.savefig(check_point)
