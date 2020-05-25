import math
import numpy as np
import ffmpeg
from scipy.fftpack import rfft, fftshift

class AudioDataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filenames):

        self.fft_size = 1024
        self.sample_count = 0
        buffer = np.array([])
        for i in range(0, len(filenames)):
            dataframe, _ = (ffmpeg
                .input(filenames[i])
                .output('-', format='s8', acodec='pcm_s8', ac=1, ar='44100')
                .overwrite_output()
                .run(capture_stdout=True)
            )
            length = math.floor(len(dataframe)/self.fft_size)#min(len(dataframe)/self.fft_size, duration*44100/self.fft_size))
            self.sample_count += length
            dataframe = np.frombuffer(dataframe, np.byte)[0: self.fft_size*length]
            buffer = np.concatenate((buffer, dataframe))

        array = []
        index = 0
        def get_ffts(waveform):
            # channel_chunks = []
            # for i in range(0, channel_count):
            
            temp_chunks = np.split(waveform, self.sample_count)

            new_chunks = []
            for j in range(0, len(temp_chunks)):
                if len(temp_chunks[j]) >= self.fft_size:# and j < max_ffts:
                    new_chunks.append(temp_chunks[j])
            # channel_chunks.append(new_chunks)

            # channel_chunk_ffts = []
            # for i in range(0, channel_count):
            # channel = channel_chunks[i]
            channel_ffts = []
            for j in range(0, len(new_chunks)):
                chunk = new_chunks[j]
                freqs = rfft(chunk)#torch.stack([chunk, torch.zeros(self.fft_size)]).permute(1, 0).fft(signal_ndim=1, normalized=False)#.ifft(signal_ndim=1)
                freqs = freqs/self.fft_size
                # freqs = fftshift(chunk)
                # freqs = freqs[0:math.floor(self.fft_size/2)]

                # freqs = np.dstack([np.real(freqs), np.imag(freqs)])[0]#.view(np.float32)[0]
                # freqs = np.concatenate(freqs)

                channel_ffts.append(freqs)
            # channel_chunk_ffts.append(predictions*.5+.5)*255(channel_ffts)

            # channel_fft_tensors = []
            # for i in range(0, channel_count):
            # channel_fft_tensors.append(torch.stack(channel_chunk_ffts[i]))
            return channel_ffts

        buffer = np.array(get_ffts(buffer))
        # import pdb; pdb.set_trace()

        # new_channels = []
        # for i in range(0, buffer.shape[0]):
        #     new_channels.append([])
        #     half_length = math.floor(buffer.shape[1]/2)
        #     for j in range(0, half_length):
        #         new_channels[i].append(complex(buffer[i][j*2], buffer[i][j*2+1]))
        # buffer =np.array(new_channels)

        # real = np.abs(buffer)#np.sqrt(np.real(buffer)*np.real(buffer)+np.imag(buffer)*np.imag(buffer))
        # imag = np.angle(buffer)

        # buffer = real+imag*1j

        # buffer[0:buffer.shape[0]-1] = buffer[1: buffer.shape[0]]-buffer[0: buffer.shape[0]-1]
        self.max_mag = max(abs(np.abs(buffer).max()), 1.)
        temp_mean = np.sum(buffer)/np.prod(buffer.shape)
        self.mean = 0#temp_mean#np.sum(buffer)/np.prod(buffer.shape)
        self.std = self.max_mag
        # buffer = buffer/self.std*.5+.5
        # self.std = math.log2(self.max_mag+1.)#np.sum(np.abs(buffer-temp_mean))/np.prod(buffer.shape)
        import matplotlib.pyplot as plt
        # real = np.log2(np.real(buffer)+1.)/self.std
        # imag = np.imag(buffer)/math.pi*.5+.5
        # buffer = real+imag*1j
        # buffer = np.sign(buffer)*np.log10(buffer+self.std*2+1.)
        # real = np.log2(np.real(buffer)+1.)/math.log2(self.std+1.)
        buffer = buffer.astype(np.float32).view(np.float32)
        # import pdb; pdb.set_trace()
        # for x in buffer:
        #     array.append([x/128.0])
        #     index += 1
        # buffer = np.array(array)
    
        self.data_train = buffer
        self.len_train  = len(self.data_train)
        self.len_train_windows = None

    def get_train_data(self, seq_len):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        length = self.len_train - seq_len
        for i in range(length):
            # print("Formatting training data: {}%".format(math.floor(i*100.0/length)))
            x, y = self._next_window(i, seq_len)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size):
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
                x, y = self._next_window(i, seq_len)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        x = window[0:window.shape[0]-1]
        y = window[window.shape[0]-1]
        return x, y
