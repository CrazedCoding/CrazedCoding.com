import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM, Conv1D, TimeDistributed, MaxPooling1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

class AudioModel():
	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):
		# self.model.add(batch_normed)
		# self.model.add(BatchNormalization())
		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				# if input_timesteps:
				# 	self.model.add(TimeDistributed(Conv1D(filters=neurons, kernel_size=1, activation='relu'), input_shape=(None, input_timesteps, input_dim)))
				# 	self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
				# 	self.model.add(TimeDistributed(Flatten()))
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		# plot_model(self.model, to_file="image.png", show_shapes=True, show_layer_names=True,rankdir='TB')

	def train(self, x, y, epochs, batch_size, save_dir):
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks,
			verbose=1
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)

	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data, verbose=1)
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				# import pdb; pdb.set_trace()
				predicted.append(self.model.predict(curr_frame[newaxis,:,:]))
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			predicted = np.array(predicted)
			prediction_seqs.append(predicted.reshape(predicted.shape[0]*predicted.shape[1], predicted.shape[2]))
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:]))
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		predicted = np.array(predicted)
		# predicted = predicted.reshape(predicted.shape[0]*predicted.shape[1], predicted.shape[2])
		return predicted

"""mport os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras import layers, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Input, concatenate, BatchNormalization,Conv1D,MaxPooling1D, Flatten, Reshape,ConvLSTM2D, TimeDistributed
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adamax
from keras.layers import Lambda
from keras.backend import slice
from keras.utils import plot_model
class AudioModel():

	def __init__(self):
		self.model = None

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):
		input_layer = Input(batch_shape=(None, configs['model']['layers'][0]["input_timesteps"], configs['model']['layers'][0]["input_dim"]), dtype=np.float32)
		current_layer = input_layer

		batch_normed = BatchNormalization(input_shape=(None, configs['model']['layers'][0]["input_timesteps"], configs['model']['layers'][0]["input_dim"]),  trainable=True) 
		current_layer = batch_normed(current_layer)
		# current_layer = MaxPooling1D(2, padding="same")(current_layer)

		# pool1 = MaxPooling1D(pool_size=math.floor(2))
		# current_layer = pool1(current_layer)

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				current_layer = Dense(neurons, activation=activation, dtype=np.float32)(current_layer)
			if layer['type'] == 'lstm':
				current_layer = LSTM(neurons, activation=activation,  input_shape=(None, input_timesteps, input_dim), dtype=np.float32, return_sequences=return_seq)(current_layer)

				# if(return_seq):

				# 	conv = Conv2D(filters=layer['neurons'],
				# 				input_shape=(None, layer['input_timesteps'], layer['neurons']), 
				# 				kernel_size=(layer['input_timesteps'], math.floor(math.log2(layer['neurons']))),
				# 				strides=(1, 1),
				# 				activation='linear',
				# 				padding='same')


				# 	import pdb; pdb.set_trace()	
				# 	current_layer = conv(Reshape(target_shape=(layer['input_timesteps'], layer['neurons']))(current_layer))
			if layer['type'] == 'dropout':
				current_layer = Dropout(dropout_rate, dtype=np.float32)(current_layer)

		self.model = Model(inputs=input_layer, output=current_layer)
		self.model.compile(loss=configs['model']['loss'], optimizer=Adamax())
		
		print('[Model] Model Compiled')
		plot_model(self.model, to_file="image.png", show_shapes=True)

	def train(self, x, y, epochs, batch_size, save_dir):
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks,
			verbose=1
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)

	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data, verbose=1)
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted
"""