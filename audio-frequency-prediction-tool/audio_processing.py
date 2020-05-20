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
from scipy.fftpack import ifft
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
	configs = json.load(open('config.json', 'r'))


	model = AudioModel()
	model.build_model(configs)

	if args.model:
		model.load_model(args.model)

	if args.inputs:
		for i in range(0, len(args.inputs)):
			data = AudioDataLoader(
				args.inputs[i],
				0.5
			)
			# in-memory training
			x, y = data.get_train_data(
				seq_len=configs['data']['sequence_length'],
				normalise=configs['data']['normalise']
			)
		
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
			# 		batch_size=configs['training']['batch_size'],
			# 		normalise=configs['data']['normalise']
			# 	),
			# 	epochs=configs['training']['epochs'],
			# 	batch_size=configs['training']['batch_size'],
			# 	steps_per_epoch=steps_per_epoch,
			# 	save_dir="./"
			# )

			# x_test, y_test = data.get_test_data(
			# 	seq_len=configs['data']['sequence_length'],
			# 	normalise=configs['data']['normalise']
			# )


	def reconstruct_waveform(ffts):
		new_channels = []
		for i in range(0, ffts.shape[0]):
			new_channels.append([])
			half_length = math.floor(ffts.shape[1]/2)
			for j in range(0, half_length):
				new_channels[i].append(complex(ffts[i][j*2], ffts[i][j*2+1]))

		new_waveform = []
		for i in range(0, ffts.shape[0]):
			
			new_waveform.append(torch.tensor(ifft(new_channels[i]).real/128.0))
			# new_waveform[i].append(new_channels[i][j].mul(2.).sub(1.).mul(max_mag).ifft(signal_ndim=1).permute(1, 0)[0])
			# new_waveform[i].append(new_channels[i][j].mul(max_freq_val.sub(min_freq_val)).add(min_freq_val).ifft(signal_ndim=1).permute(1, 0)[0])
			
		return torch.cat(new_waveform)

	if args.targets and args.output:


		target_data = []
		
		target_xy = []
		
		min_x_size = 1E32

		max_std = -1E32

		for i in range(0, len(args.targets)): 
			target_data.append(AudioDataLoader(
				args.targets[i],
				configs['data']['train_test_split']
			))
			target_x, target_y = target_data[i].get_train_data(
				seq_len=configs['data']['sequence_length'],
				normalise=configs['data']['normalise']
			)
			if target_data[i].std > max_std:
				max_std = target_data[i].std

			target_xy.append([target_x, target_y])
			if len(target_x) < min_x_size:
				min_x_size = len(target_x)

		def weight_function(i, j):
		 	return (math.cos(i*1.0/min_x_size*math.pi*4*(1.+j*1.0/len(args.targets)))*.25+.75)

		final_target = []
		for i in range(0, min_x_size):
			final_target.append(target_xy[0][0][i]*0.0)
			for j in range(0, len(args.targets)): 
				total_weight = 0
				for k in range(0, len(args.targets)): 
					total_weight += weight_function(i, k)/len(args.targets)
				weight = weight_function(i, j)/total_weight
				print("Weight: {}".format(weight))
				final_target[i] += target_xy[j][0][i]*weight

		predictions = model.predict_point_by_point(np.array(final_target))
		#import pdb; pdb.set_trace()
			
		final_waveform = reconstruct_waveform((predictions)*max_std)

		torchaudio.save(args.output, torch.clamp((final_waveform).cpu().detach(), -1, 1), 44100, precision=32)

	# plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
	# plot_results(np.floor((np.transpose(predictions, (1, 0))*.5+.5)*255), np.floor((np.transpose(y, (1, 0))*.5+.5)*255))


if __name__ == '__main__':
	main()