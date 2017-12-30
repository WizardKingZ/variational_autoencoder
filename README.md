# variational_autoencoder

## Final Project Notebook.ipynb 

This is the finalised jupyter notebook that clearly displays our code and plots. Note if you wanna run the code again, you need to change the parts of trained=True to trained=False. This is to avoid redundant training time. 

----------------------------------------------------------------------------------------------------------

"python": directory for source code folder. 

	custom_plot.py: customized plotting library.

	dataset.py: base class for dataset that mimics the same functionality as mnist from tensorflow.

	svhn.py: class that inherits the dataset class in dataset.py, interfacing with svhn data.

	vae.py: main class for variational autoencoder implementation
