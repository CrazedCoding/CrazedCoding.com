import math
import numpy as np
import ffmpeg
from scipy.fftpack import fft

class AudioDataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split):
        dataframe, _ = (ffmpeg
            .input(filename)
            .output('-', format='s8', acodec='pcm_s8', ac=1, ar='44100')
            .overwrite_output()
            .run(capture_stdout=True)
        )
        fft_size = 32
        duration = 32
        sample_count = math.floor(len(dataframe)/fft_size)#min(len(dataframe)/fft_size, duration*44100/fft_size))
        buffer = np.frombuffer(dataframe, np.byte)[0: fft_size*sample_count]
        # buffer = np.dstack((range(0, len(buffer)), buffer))[0]

        array = []
        index = 0
        def get_ffts(waveform):
            # channel_chunks = []
            # for i in range(0, channel_count):
            
            temp_chunks = np.split(waveform, sample_count)

            new_chunks = []
            for j in range(0, len(temp_chunks)):
                if len(temp_chunks[j]) >= fft_size:# and j < max_ffts:
                    new_chunks.append(temp_chunks[j])
            # channel_chunks.append(new_chunks)

            # channel_chunk_ffts = []
            # for i in range(0, channel_count):
            # channel = channel_chunks[i]
            channel_ffts = []
            for j in range(0, len(new_chunks)):
                chunk = new_chunks[j]
                freqs = fft(chunk)#torch.stack([chunk, torch.zeros(fft_size)]).permute(1, 0).fft(signal_ndim=1, normalized=False)#.ifft(signal_ndim=1)
                freqs = np.dstack([freqs.real, freqs.imag])[0]#.view(np.float32)[0]
                freqs = np.concatenate(freqs)
                channel_ffts.append(freqs)
            # channel_chunk_ffts.append(predictions*.5+.5)*255(channel_ffts)

            # channel_fft_tensors = []
            # for i in range(0, channel_count):
            # channel_fft_tensors.append(torch.stack(channel_chunk_ffts[i]))
            return channel_ffts
        buffer = np.array(get_ffts(buffer))
        # buffer[0:buffer.shape[0]-1] = buffer[1: buffer.shape[0]]-buffer[0: buffer.shape[0]-1]
        self.max_mag = max(abs(np.abs(buffer).max()), 1.)
        temp_mean = np.sum(buffer)/np.prod(buffer.shape)
        self.mean = 0#temp_mean#np.sum(buffer)/np.prod(buffer.shape)
        self.std = self.max_mag#np.sum(np.abs(buffer-temp_mean))/np.prod(buffer.shape)
        import matplotlib.pyplot as plt
        buffer = (buffer-self.mean)/self.std

        # for x in buffer:
        #     array.append([x/128.0])
        #     index += 1
        # buffer = np.array(array)
    
        i_split = int(len(buffer) * split)
        self.data_train = buffer[:i_split]
        self.data_test  = buffer[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)

        x = data_windows[:, :-1]
        y = data_windows[:, -1, ]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        length = self.len_train - seq_len
        for i in range(length):
            print("Formatting training data: {}%".format(math.floor(i*100.0/length)))
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        x = window[0:window.shape[0]-1]
        y = window[window.shape[0]-1]
        return x, y
