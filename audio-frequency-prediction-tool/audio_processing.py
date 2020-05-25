__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import time
import math
import json
import matplotlib.pyplot as plt
from audio_data import AudioDataLoader
from audio_model import AudioModel
import numpy as np
from scipy.fftpack import irfft
import torchaudio, torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputs", nargs='+', required=False, help="Paths to audio files to train from.")
parser.add_argument("-m", "--model", required=False, help="Path to model file to load.")
parser.add_argument("-t", "--targets", nargs='+', required=False, help="Paths to audio files to try to predict.")
parser.add_argument("-o", "--output", required=False, help="Paths to audio output file.")
args = parser.parse_args()

def plot_results(predicted_data, true_data):
	fig = plt.figure(facecolor='white')
	plt.imshow(true_data, cmap='gray')
	fig = plt.figure(facecolor='white')
	plt.imshow(predicted_data, cmap='gray')
	plt.legend()
	plt.show()

def main():

	print("Loading config.json settings file.")
	configs = json.load(open('config.json', 'r'))

	print("Building NN model.")
	model = AudioModel()
	model.build_model(configs)

	if args.model:
		print("Loading model weights file: {}".format(args.model))
		model.load_model(args.model)

	if args.inputs:
		print("Loading audio training input file(s).")
		# for i in range(0, len(args.inputs)):
		print("Loading audio training input file: {}".format(args.inputs))
		data = AudioDataLoader(
			args.inputs
		)
		print("Reformatting audio data.")
		# in-memory training
		x, y = data.get_train_data(
			seq_len=configs['data']['sequence_length']
		)
	
		print("Training model with audio data.")
		model.train(
			x,
			y,
			epochs = configs['training']['epochs'],
			batch_size = configs['training']['batch_size'],
			save_dir="./"
		)

			# out-of memory generative training
			# steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])

			# model.train_generator(
			# 	data_gen=data.generate_train_batch(
			# 		seq_len=configs['data']['sequence_length'],
			# 		batch_size=configs['training']['batch_size']
			# 	),
			# 	epochs=configs['training']['epochs'],
			# 	batch_size=configs['training']['batch_size'],
			# 	steps_per_epoch=steps_per_epoch,
			# 	save_dir="./"
			# )

			# x_test, y_test = data.get_test_data(
			# 	seq_len=configs['data']['sequence_length']
			# )



	target_data = []
	
	target_xy = []
	
	min_x_size = 1E32

	max_std = -1E32
	fft_size = None

	def reconstruct_waveform(rffts):
		new_waveform = []

		for i in range(0, rffts.shape[0]):

			# mag = np.exp2(np.real(ffts[i])*max_std)-1.
			# phase = (np.imag(ffts[i])*2.-1.	)*math.pi
			
			# freqs = mag * np.exp( 1j * phase)

			# freqs = np.concatenate(freqs, freqs)
			# freqs = ifftshift(freqs)

			freqs = rffts[i]*fft_size


			# Or if you want the real and imaginary complex components separately,
			# freqs = mag * np.cos(phase) + mag * np.sin(phase)*1j
			print("Max mag {}".format(freqs.max()))
			# print("Max angle {}".format(phase.max()))
			# import pdb; pdb.set_trace()
			

			new_waveform.append(torch.tensor(irfft(freqs)/128.0))
			# new_waveform[i].append(new_channels[i][j].mul(2.).sub(1.).mul(max_mag).ifft(signal_ndim=1).permute(1, 0)[0])
			# new_waveform[i].append(new_channels[i][j].mul(max_freq_val.sub(min_freq_val)).add(min_freq_val).ifft(signal_ndim=1).permute(1, 0)[0])
			
		return torch.cat(new_waveform)

	if args.targets and args.output:

		print("Loading target audio files for point-by-point prediction.")
		for i in range(0, len(args.targets)): 
			print("Loading target audio file: {}.".format(args.targets[i]))
			target_data.append(AudioDataLoader(
				[args.targets[i]]
			))

			fft_size = target_data[i].fft_size
			target_x, target_y = target_data[i].get_train_data(
				seq_len=configs['data']['sequence_length']
			)
			if target_data[i].std > max_std:
				max_std = target_data[i].std

			target_xy.append([target_x, target_y])
			if len(target_x) < min_x_size:
				min_x_size = len(target_x)

		def weight_function(i, j):
		 	return (math.cos(i*1.0/min_x_size*32.*math.pi**(1.+j*1.0/len(args.targets)))*.5+.5)

		final_target = []
		print("Merging target audio files with randomly sinusoidally changing weights.")
		for i in range(0, min_x_size):
			final_target.append(target_xy[0][0][i]*0.0)
			for j in range(0, len(args.targets)): 
				total_weight = 0
				for k in range(0, len(args.targets)): 
					total_weight += weight_function(i, k)/len(args.targets)
				weight = weight_function(i, j)/total_weight
				print("Weight: {}".format(weight))
				final_target[i] += target_xy[j][0][i]*weight

		print("Using current NN model to do point-by-point prediction of the merged target audio files.")
		predictions = model.predict_point_by_point(np.array(final_target))
		# predictions = model.predict_sequences_multiple(np.array(final_target), configs['data']['sequence_length'], configs['data']['sequence_length'])
		# predictions = model.predict_sequence_full(np.array(final_target)[0:math.floor(len(final_target)/2)], configs['data']['sequence_length'])
		# predictions = np.array(predictions)
		# predictions = predictions.reshape(predictions.shape[0]*predictions.shape[1], predictions.shape[2])
		print("Constructing waveform out of final point-by-point prediction.")
		# final_waveform = reconstruct_waveform(np.array(target_xy[0][1]).astype(np.float32).view(np.complex64))
		predictions = predictions.astype(np.float32).view(np.float32)
		# predictions = predictions.reshape(math.floor(predictions.shape[0]/target_data[0].fft_size), target_data[0].fft_size)
		final_waveform = reconstruct_waveform(predictions)

		print("Saving/encoding audio file to file: {}".format(args.output))
		torchaudio.save(args.output, torch.clamp((final_waveform).cpu().detach(), -1, 1), 44100, precision=32)

	# plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
	# plot_results(np.floor((np.transpose(predictions, (1, 0))*.5+.5)*255), np.floor((np.transpose(y, (1, 0))*.5+.5)*255))


if __name__ == '__main__':
	main()