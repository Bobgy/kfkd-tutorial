"""
To use this script, first run this to fit your first model:

  python kfkd.py fit

Then train a bunch of specialists that intiliaze their weights from
your first model:

  python kfkd.py fit_specialists net.pickle

Plot their error curves:

  python kfkd.py plot_learning_curves net-specialists.pickle

And finally make predictions to submit to Kaggle:

  python kfkd.py predict net-specialists.pickle
"""

import cPickle as pickle
from datetime import datetime
import os
import sys

from matplotlib import pyplot
import numpy as np
import lasagne
from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
import theano.tensor as T

import time

try:
	from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
	from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer


sys.setrecursionlimit(10000)  # for pickle...
#np.random.seed(42)

FTRAIN = '~/data/kaggle-facial-keypoint-detection/training.csv'
FTEST = '~/data/kaggle-facial-keypoint-detection/test.csv'
FLOOKUP = '~/data/kaggle-facial-keypoint-detection/IdLookupTable.csv'


def float32(k):
	return np.cast['float32'](k)


def load(test=False, cols=None):
	"""Loads data from FTEST if *test* is True, otherwise from FTRAIN.
	Pass a list of *cols* if you're only interested in a subset of the
	target columns.
	"""
	fname = FTEST if test else FTRAIN
	df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

	# The Image column has pixel values separated by space; convert
	# the values to numpy arrays:
	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	if cols:  # get a subset of columns
		df = df[list(cols) + ['Image']]

	print(df.count())  # prints the number of values for each column
	df = df.dropna()  # drop all rows that have missing values in them

	X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
	X = X.astype(np.float32)

	if not test:  # only FTRAIN has any target columns
		y = df[df.columns[:-1]].values
		y = (y - 48) / 48  # scale target coordinates to [-1, 1]
		X, y = shuffle(X, y, random_state=42)  # shuffle train data
		y = y.astype(np.float32)
	else:
		y = None

	return X, y


def load2d(test=False, cols=None):
	X, y = load(test=test, cols=cols)
	X = X.reshape(-1, 1, 96, 96)
	return X, y


def plot_sample(x, y, axis):
	img = x.reshape(96, 96)
	axis.imshow(img, cmap='gray')
	if y is not None:
		axis.scatter(y[0::2], y[1::2], marker='x', s=10)


def plot_weights(weights):
	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust(
		left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		ax.imshow(weights[:, i].reshape(96, 96), cmap='gray')
	pyplot.show()


class FlipBatchIterator(BatchIterator):
	flip_indices = [
		(0, 2), (1, 3),
		(4, 8), (5, 9), (6, 10), (7, 11),
		(12, 16), (13, 17), (14, 18), (15, 19),
		(22, 24), (23, 25),
		]

	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

		# Flip half of the images in this batch at random:
		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs / 2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1]

		if yb is not None:
			# Horizontal flip of all x coordinates:
			yb[indices, ::2] = yb[indices, ::2] * -1

			# Swap places, e.g. left_eye_center_x -> right_eye_center_x
			for a, b in self.flip_indices:
				yb[indices, a], yb[indices, b] = (
					yb[indices, b], yb[indices, a])

		return Xb, yb


class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

		epoch = train_history[-1]['epoch']
		new_value = np.cast['float32'](self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
	def __init__(self, patience=100):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0
		self.best_weights = None

	def __call__(self, nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(
				self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()

class FactoredLayer(layers.Layer):
	def __init__(self, incoming, num_units, num_hidden, W, b,
				nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
		super(FactoredLayer, self).__init__(incoming, **kwargs)
		self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
							 else nonlinearity)
		num_inputs = int(np.prod(self.input_shape[1:]))
		self.num_units = num_units
		u, s, v = np.linalg.svd(W)
		self.W0 = T.constant(u[:, :num_hidden])
		self.W1 = self.create_param(
					np.dot(np.diag(s[:num_hidden]), v[:num_hidden]),
					(num_hidden, num_units), name="W1")
		self.b = (self.create_param(b, (num_units,), name="b")
			if b is not None else None)

	def get_output_for(self, input, **kwargs):
		if input.ndim > 2:
			input = input.flatten(2)

		activation = T.dot(input, self.W0).dot(self.W1)
		if self.b is not None:
			activation = activation + self.b.dimshuffle('x', 0)
		return self.nonlinearity(activation)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.num_units)


net = NeuralNet(
	layers=[
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		('dropout1', layers.DropoutLayer),
		('hidden2', layers.DenseLayer),
		('dropout2', layers.DropoutLayer),
		('hidden3', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
	input_shape=(None, 1, 9216),
	hidden1_num_units=1000,
	dropout1_p=0.2,
	hidden2_num_units=1000,
	dropout2_p=0.4,
	hidden3_num_units=1000,
	output_num_units=30, output_nonlinearity=None,

	update_learning_rate=theano.shared(float32(0.03)),
	update_momentum=theano.shared(float32(0.9)),

	regression=True,
	batch_iterator_train=FlipBatchIterator(batch_size=128),
	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.999),
		EarlyStopping(patience=200),
		],
	max_epochs=3000,
	verbose=1,
	)


def fit():
	X, y = load()
	net.fit(X, y)
	with open('net.pickle', 'wb') as f:
		pickle.dump(net, f, -1)


from collections import OrderedDict

from sklearn.base import clone

def predict(fname='net.pickle'):
	with open(fname, 'rb') as f:
		net = pickle.load(f)

	X = load(test=True)[0]
	y_pred = np.empty((X.shape[0], 0))

	y_pred = net.predict(X)

	y_pred2 = y_pred * 48 + 48
	y_pred2 = y_pred2.clip(0, 96)

	columns = ( 'left_eye_center_x',	'left_eye_center_y',
				'right_eye_center_x',	'right_eye_center_y',
				'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
				'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
				'right_eye_inner_corner_x','right_eye_inner_corner_y',
				'right_eye_outer_corner_x','right_eye_outer_corner_y',
				'left_eyebrow_inner_end_x','left_eyebrow_inner_end_y',
				'left_eyebrow_outer_end_x','left_eyebrow_outer_end_y',
				'right_eyebrow_inner_end_x','right_eyebrow_inner_end_y',
				'right_eyebrow_outer_end_x','right_eyebrow_outer_end_y',
				'nose_tip_x','nose_tip_y',	'mouth_left_corner_x',
				'mouth_left_corner_y',	'mouth_right_corner_x',
				'mouth_right_corner_y',	'mouth_center_top_lip_x',
				'mouth_center_top_lip_y',
				'mouth_center_bottom_lip_x',
				'mouth_center_bottom_lip_y' )

	df = DataFrame(y_pred2, columns=columns)

	lookup_table = read_csv(os.path.expanduser(FLOOKUP))
	values = []

	for index, row in lookup_table.iterrows():
		values.append((
			row['RowId'],
			df.ix[row.ImageId - 1][row.FeatureName],
			))

	now_str = datetime.now().isoformat().replace(':', '-')
	submission = DataFrame(values, columns=('RowId', 'Location'))
	filename = 'submission-{}.csv'.format(now_str)
	submission.to_csv(filename, index=False)
	print("Wrote {}".format(filename))


def fit_net2(fname='net.pickle', sfname='net2.pickle'):
	with open(fname, 'r') as f:
		net = pickle.load(f)
	l1=net.get_all_layers()
	net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('hidden1', layers.FactoredLayer),
			('dropout1', layers.DropoutLayer),
			('hidden2', layers.FactoredLayer),
			('dropout2', layers.DropoutLayer),
			('hidden3', layers.FactoredLayer),
			('output', layers.DenseLayer),
			],
		input_shape=(None, 1, 9216),
		hidden1_num_units=1000,
		hidden1_num_hidden=100,
		dropout1_p=0.2,
		hidden2_num_units=1000,
		hidden2_num_hidden=100,
		dropout2_p=0.4,
		hidden3_num_units=1000,
		hidden3_num_hidden=100,
		output_num_units=30, output_nonlinearity=None,

		update_learning_rate=theano.shared(float32(0.03)),
		update_momentum=theano.shared(float32(0.9)),

		regression=True,
		batch_iterator_train=FlipBatchIterator(batch_size=128),
		on_epoch_finished=[
			AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
			AdjustVariable('update_momentum', start=0.9, stop=0.999),
			EarlyStopping(patience=200),
			],
		max_epochs=1,
		verbose=1,
	)
	
	X, y = load()
	net2.fit(X, y)
	net2.load_params_from(net.get_all_params_values())
	#net2.fit(X, y)
	"""
	l2=net2.get_all_layers()
	print(l2)
	for i in xrange(len(l1)):
		if i!=10 and i!=12:
			all_param_values = lasagne.layers.get_all_param_values(l1[i])
			lasagne.layers.set_all_param_values(l2[i], all_param_values)
	"""
	with open(sfname, 'wb') as f:
		pickle.dump(net2, f, -1)


def rebin( a, newshape ):
	from numpy import mgrid
	assert len(a.shape) == len(newshape)

	slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
	coordinates = mgrid[slices]
	indices = coordinates.astype('i')   #choose the biggest smaller integer index
	return a[tuple(indices)]


def plot_learning_curves(fname='net.pickle'):
	with open(fname, 'r') as f:
		model = pickle.load(f)

	fig = pyplot.figure(figsize=(10, 6))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_color_cycle(
		['c', 'c', 'm', 'm', 'y', 'y', 'k', 'k', 'g', 'g', 'b', 'b'])

	valid_losses = []
	train_losses = []

	valid_loss = np.array([i['valid_loss'] for i in model.train_history_])
	train_loss = np.array([i['train_loss'] for i in model.train_history_])
	valid_loss = np.sqrt(valid_loss) * 48
	train_loss = np.sqrt(train_loss) * 48

	valid_loss = rebin(valid_loss, (100,))
	train_loss = rebin(train_loss, (100,))

	valid_losses.append(valid_loss)
	train_losses.append(train_loss)
	ax.plot(valid_loss,
			label='Original Validation', linewidth=3)
	ax.plot(train_loss,
			label='Original Training', linestyle='--', linewidth=3, alpha=0.6)
	ax.set_xticks([])

	ax.legend()
	ax.set_ylim((1.0, 4.0))
	ax.grid()
	pyplot.ylabel("RMSE")
	pyplot.show()

def plot_image(fname='net.pickle', offset=32):

	with open(fname, 'rb') as f:
		net = pickle.load(f)

	X = load(test=True)[0]
	for i in xrange(4):
		X = np.vstack([X, X])
	print('testing set shape: {}'.format(X.shape))

	start_time = time.time();
	print('Start predicting.');
	y_pred = net.predict(X)
	print('Prediction made, %.4f seconds elapsed.' % (time.time()-start_time))

	y_pred2 = y_pred * 48 + 48
	y_pred2 = y_pred2.clip(0, 96)

	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust(
		left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
		plot_sample(X[32+i], y_pred2[32+i], ax)

	pyplot.show()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(__doc__)
	else:
		func = globals()[sys.argv[1]]
		func(*sys.argv[2:])
