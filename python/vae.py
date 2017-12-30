import numpy as np
import tensorflow as tf

class VAE:
	""" 
	The Variational Bayes Autoencoder (https://arxiv.org/pdf/1312.6114.pdf) 

	We used Gaussian encoder and Gaussian/Bernoulli decoder. The encoder network is either convolutional or simple feed-forward
	The decoder network is simple feed-forward

	Referred to the following notebook. 
	"http://www.cvc.uab.es/people/joans/slides_tensorflow/tensorflow_html/vae-Jan-Hendrik-Metzen.html" 

	See Kingma, Welling, "Auto-Encoding Variational Bayes" 
	"""

	def __init__(self, encode_nk_arch, decode_nk_arch, 
				learn_rate=0.001, mini_batch_size=100,
				verbose=True, step=5, epochs=10, opt='adam',
				decoder_type='bern'):
		"""
		Parameters:
		encode_nk_arch: the network architecture of the encoding network; dictionary
		e.g. 
		if we treat each image data as the single one-d vector, then the input size will be specified as an integer
		if we supply RGB images, the input size will be a list. In this case, we are expecting convolutional layers 
		For flat structure, we can have the following input
		encode_nk_arch = {'input_size': 28*28,
                  'layers': [{'layer_name': 'layer_1', 'hidden_units': 100, 'activation_function': tf.nn.relu, 'layer_type': 'full_conn'},
                            {'layer_name': 'layer_2', 'hidden_units': 20, 'activation_function': tf.nn.relu, 'layer_type': 'full_conn'}]}
        For a RGB structure, our input structure should look like the following
        users should not use maxpooling because it is not invertible. ideally, you would want to keep the decoder as the mirror image of the encoder
        encode_nk_arch = {'input_size': [28, 28, 1],
        				  'layers': [{'layer_name': 'layer_1', 'layer_type': 'conv', 'filter': [5, 5, 1], 'num_activation_map': 64, 'activation_function': tf.nn.relu},
									 'layer_name': 'layer_2', 'layer_type': 'full_conn', 'hidden_units': 20, 'activation_function': tf.nn.relu}]}
        				  }]
		}
		
		Note the last layer should represent the number of latent variables
		decode_nk_arch: the network architecture of the decoding network; dictionary
		e.g. decode_nk_arch = {'layers': [{'layer_name': 'layer_1', 'hidden_units': 100, 'activation_function': tf.nn.relu}]}
		learn_rate: learning rate used in optimization procedures, default is 0.001
		mini_batch_zize: the mini batch size, default is 100 
		verbose: should the program produce training output, default True
		step: how many number episodes (epochs) to print; default is 5
		opt: the optimization used in tensorflow; default is adam 
		decoder_type: It is either bernoulli (bern) or normal (norm)
		"""
		tf.reset_default_graph()
		self.encode_nk_arch = encode_nk_arch
		self.decode_nk_arch = decode_nk_arch
		self.num_latent_var = encode_nk_arch['layers'][-1]['hidden_units']
		self.input_size = encode_nk_arch['input_size']
		if type(self.input_size) == int:
			self.input_type = 'bw'
		elif type(self.input_size) == list:
			self.input_type = 'rgb'
			self.reshape_conv = [-1]
			self.reshape_conv.extend(self.input_size)
		self.learn_rate = learn_rate
		self.mini_batch_size = mini_batch_size
		self.verbose = verbose
		self.step = step
		self.epochs = epochs
		self.opt = opt
		self.small_val = 1e-10
		self.decoder_type = decoder_type


		## X is the image data
		self.X = tf.placeholder(tf.float32, [None, np.prod(self.encode_nk_arch['input_size'])])

		## create the encoding and decoding networks
		self.__create_network()

		## Define loss function and the optimizer
		self.__create_loss_opt()

		## Initialize the TensorFlow global variables
		init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		## Create the TensorFlow sessions
		self.sess = tf.Session()
		self.sess.run(init)


	# ------------------------------ stuff from class ------------------------------------------- #
	# cnn conv stuff
	def __conv(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def __maxpool(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def __norm(self, x): 
		return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)

	#----------------------------- end -------------------------------------------- #

	def __create_network(self):
		"""
		Build the network
		"""
		encode_nk_weights = self.__init_weights(self.encode_nk_arch['layers'])
		self.z_mu, self.z_log_sigma_squared = self.__encode_param(encode_nk_weights, self.encode_nk_arch['layers'])

		## Draw one sample z from Gaussian distribution
		eps = tf.random_normal((self.mini_batch_size, self.num_latent_var), 0, 1,
								dtype=tf.float32)
		## z = mu + sigma*epsilon
		## use the standard normal distribution 
		self.z = tf.add(self.z_mu, 
						tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_squared)),
						eps))

		## create decoder network
		decode_nk_weights = self.__init_weights(self.decode_nk_arch['layers'], type='decode')
		self.X_mu, self.X_log_sigma_squared = self.__decode_param(decode_nk_weights, self.decode_nk_arch['layers'])


	## initialize weights stuff .......
	def __init_weights(self, layers, type='encode'):
		weights = {}
		n_layers = len(layers)
		## set up layer by layer
		for i in range(n_layers):
			layer = layers[i]
			temp = {}
			if i == 0:
				if type=='encode':
					init_in = self.input_size
					if self.input_type == 'rgb':
						init_in = layer['filter']
						init_in.append(layer['num_activation_map'])
				else:
					init_in = self.num_latent_var
				if layer['layer_type'] == 'conv':
					temp['weights']=self.__init_tf_var(init_in,  
													None,
												   	type+layer['layer_name'],
												   	var_type='cnn')
					temp['bias']=self.__init_tf_var(layer['num_activation_map'], 
													None, 
													type+layer['layer_name'],
													var_type='bias')
				elif layer['layer_type'] == 'full_conn':
					temp['weights']=self.__init_tf_var(init_in, 
												   	layer['hidden_units'], 
												   	type+layer['layer_name'])
					temp['bias']=self.__init_tf_var(layer['hidden_units'], 
													None, 
													type+layer['layer_name'],
													var_type='bias')
			else:
				prev_layer = layers[i-1]
				if layer['layer_type'] == 'conv':
					filter_shape = layer['filter']
					filter_shape.append(layer['num_activation_map'])
					temp['weights']=self.__init_tf_var(filter_shape, 
												   None,
												   type+layer['layer_name'], 
												   var_type='cnn')
					temp['bias']=self.__init_tf_var(layer['num_activation_map'], 
												None, 
											    type+layer['layer_name'],
											    var_type='bias')
				elif layer['layer_type'] == 'full_conn':
					if prev_layer['layer_type'] == 'conv':
						temp['weights']=self.__init_tf_var(np.prod(self.input_size[:2])*prev_layer['num_activation_map'], 
												   	layer['hidden_units'], 
												   	type+layer['layer_name'])
					elif prev_layer['layer_type'] == 'full_conn':
						temp['weights']=self.__init_tf_var(prev_layer['hidden_units'], 
												   	layer['hidden_units'], 
												   	type+layer['layer_name'])
					temp['bias']=self.__init_tf_var(layer['hidden_units'], 
												None, 
											    type+layer['layer_name'],
											    var_type='bias')
			weights[layer['layer_name']] = temp
		if type=='encode':
			param_size = self.num_latent_var
		else: 
			param_size = np.prod(self.input_size)
		m = {}
		m['weights'] = self.__init_tf_var(layers[-1]['hidden_units'],
										  param_size,
										  type+"mean_weight")
		m['bias'] = self.__init_tf_var(param_size,
									   None,
									   type+"mean_bias",
									   var_type='bias')
		weights['mean'] = m
		var = {}
		var['weights'] = self.__init_tf_var(layers[-1]['hidden_units'],
											param_size,
											type+"log_sigma_squared_weight")
		var['bias'] = self.__init_tf_var(param_size,
										 None,
										 type+"log_sigma_squared_bias",
										 var_type='bias')
		weights['log_sigma_squared'] = var
		return weights

	def __init_tf_var(self, n_in, n_out, layer_name, var_type='weight'):
		if var_type == 'weight':
			return tf.get_variable(layer_name+var_type, shape=[n_in, n_out])
		elif var_type == 'bias':
			return tf.get_variable(layer_name+var_type, shape=[n_in])
		elif var_type == 'cnn':
			return tf.get_variable(layer_name+var_type, shape=n_in)

	def __encode_param(self, weights, layers):
		n_layers = len(layers) 
		l = []
		for i in range(n_layers):
			if layers[i]['layer_type']=='conv': 
				mul_func = self.__conv
			elif layers[i]['layer_type']=='full_conn':
				mul_func = tf.matmul
			if i == 0:
				image = self.X
				if layers[i]['layer_type']=='conv': 
					image = tf.reshape(self.X, self.reshape_conv)
				l.append(layers[i]['activation_function'](tf.add(mul_func(image, weights[layers[i]['layer_name']]['weights']),
																 weights[layers[i]['layer_name']]['bias'])))
			else:
				temp_layer = l[i-1]
				if layers[i-1]['layer_type'] == 'conv':
					## need to flatten 
					temp_layer = tf.reshape(temp_layer, [-1, np.prod(self.input_size[:2])*layers[i-1]['num_activation_map']])

				l.append(layers[i]['activation_function'](tf.add(mul_func(temp_layer, weights[layers[i]['layer_name']]['weights']),
																 	weights[layers[i]['layer_name']]['bias'])))
		## here we suppose the prior distribution of the latent model is a standard Gussian 
		## in this case, we can typically assume the variational inference family with variance as a diagonal matrix
		z_mu = tf.add(tf.matmul(l[-1], weights['mean']['weights']),
					  weights['mean']['bias'])
		z_log_sigma_squared = tf.add(tf.matmul(l[-1], weights['log_sigma_squared']['weights']),
					  weights['log_sigma_squared']['bias'])
		return z_mu, z_log_sigma_squared


	def __decode_param(self, weights, layers):
		n_layers = len(layers) 
		l = []
		for i in range(n_layers):
			if layers[i]['layer_type']=='conv': 
				mul_func = self.__conv
			elif layers[i]['layer_type']=='full_conn':
				mul_func = tf.matmul
			if i == 0:
				l.append(layers[i]['activation_function'](tf.add(tf.matmul(self.z, weights[layers[i]['layer_name']]['weights']),
																 weights[layers[i]['layer_name']]['bias'])))
			else:
				l.append(layers[i]['activation_function'](tf.add(tf.matmul(l[i-1], weights[layers[i]['layer_name']]['weights']),
																 weights[layers[i]['layer_name']]['bias'])))
		X_log_sigma_squared = None
		if self.decoder_type == 'bern':
			X_mu = tf.nn.sigmoid(tf.add(tf.matmul(l[-1], weights['mean']['weights']), 
                                 weights['mean']['bias']))
		elif self.decoder_type == 'norm':
			## force the mean to between 0 and 1
			X_mu = tf.nn.sigmoid(tf.add(tf.matmul(l[-1], weights['mean']['weights']),
					  weights['mean']['bias']))
			X_log_sigma_squared = tf.add(tf.matmul(l[-1], weights['log_sigma_squared']['weights']),
					  weights['log_sigma_squared']['bias']) 
		return X_mu, X_log_sigma_squared

	def __create_loss_opt(self):
		## The loss consists of two part:
		## The encoding part and the decoding part
		## reconstruction loss = x log(p) + x log(1-p)
		## add a small value to make sure we can do log, here it is 1e-10
		if self.decoder_type == 'bern':
			decode_loss = tf.reduce_sum(self.X 
								  	  * tf.log(self.small_val + self.X_mu)
								  	  + (1 - self.X ) 
								      * tf.log(self.small_val + 1 - self.X_mu),
					                  1)
		elif self.decoder_type == 'norm':
		## this is the same experession as encode loss
			decode_loss = -0.5 * tf.reduce_sum(tf.square(self.X-self.X_mu)
											/ tf.exp(self.X_log_sigma_squared)
											+ 2*np.pi*tf.exp(self.X_log_sigma_squared), 1)
		encode_loss = 0.5 * tf.reduce_sum(self.z_log_sigma_squared
                                           + tf.square(self.z_mu) 
                                           - self.z_log_sigma_squared
                                           - 1, 1)
		self.cost = tf.reduce_mean(encode_loss - decode_loss)   # average over batch
		# Use ADAM optimizer
		if self.opt == 'adam':
			self.optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cost)
		elif self.opt == 'sgd':
			self.optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cost)
		elif self.opt == 'rms':
			self.optimizer = tf.train.RMSPropOptimizer(self.learn_rate).minimize(self.cost)

	def __mini_batch_train(self, train_data):
		"""
		train mini-batch
		return the cost of the mini-batch
		"""
		opt, cost = self.sess.run((self.optimizer, self.cost),
								  feed_dict={self.X: train_data})
		return cost

	def recreate(self, sample):
		return self.sess.run(self.X_mu, 
							  feed_dict={self.X: sample})

	def train(self, train_data, size, type='mnist'):
		"""
		mini_batch training 
		"""
		for epoch in range(self.epochs):
			avg_cost = 0.
			total_batch = int(size / self.mini_batch_size)
			for i in range(total_batch):
				batch_xs, _ = train_data.next_batch(self.mini_batch_size)
				cost = self.__mini_batch_train(batch_xs)
				avg_cost += cost / size * self.mini_batch_size
			if epoch % self.step == 0:
				print("Epoch:", '%04d' % (epoch+1),
					  "cost=", "{:.9f}".format(avg_cost))

	def generate(self, z_mu):
		""" 
		generate data by sampling from the latent space
		"""
		return self.sess.run(self.X_mu,
							 feed_dict={self.z: z_mu})

	def transform(self, X):
		""" 
		map the data X into the latent space
		"""
		return self.sess.run(self.z_mu, feed_dict={self.x: X})

	def save(self, check_point = 'model.ckpt'): 
		## save the model
		save_path = self.saver.save(self.sess, check_point)

	def load(self, check_point= 'model.ckpt'):
		## load the model
		self.saver.restore(self.sess, check_point)

		