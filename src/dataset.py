#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
from absl import flags, logging
from pathlib import Path
import numpy as np
import torch 
from torch.utils.data import Dataset, ConcatDataset
import MelFilterBank as mel
from matplotlib import pyplot as plt
import IPython
from scipy import hanning
from stft import STFT
from librosa.filters import mel as librosa_mel_fn
from librosa.core import resample
from mne.filter import filter_data

from scipy.signal import hilbert, decimate
from scipy.fftpack import next_fast_len
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression

import os
from pynwb import NWBHDF5IO
import scipy

# flags.DEFINE_string('data_dir', '/hdd/hyy/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS', '')
# flags.DEFINE_string('data_dir', '/hdd/hyy/SentenceProductionDutch', '')

flags.DEFINE_string('dataset', 'word', '')
flags.DEFINE_string('patient', 'sub-01','')

# flags.DEFINE_string('dataset', 'sentence', '')
# flags.DEFINE_string('patient', 'p1','')

# flags.DEFINE_boolean('patient_eight', False,'')
# flags.DEFINE_boolean('patient_thirteen', False,'')

# flags.DEFINE_float('window_size', 400., 'in ms')
# flags.DEFINE_float('window_size', 394.74, 'in ms')
flags.DEFINE_float('window_size', 464.4, 'in ms')
flags.DEFINE_integer('sampling_rate_eeg', 1024, '')
flags.DEFINE_float('versatz_windows', 0.576, '')
# flags.DEFINE_float('versatz_windows', 1., '')
flags.DEFINE_integer('num_audio_classes', 256, '')#ignore


# flags.DEFINE_float('train_test_split', .90, '')#1.5min?
flags.DEFINE_float('train_split', .80, '')
flags.DEFINE_float('val_split', .10, '')
flags.DEFINE_boolean('use_MFCCs', True,'')#why MFCCs? not Mel-spec?
flags.DEFINE_boolean('double_trouble', False,'')

FLAGS = flags.FLAGS

def mu_law(x): #for audio conversion from regression to classification
    return np.sign(x)*np.log(1+255*np.abs(x))/np.log(1+255)

def mu_law_inverse(x):
    return np.sign(x)*(1./255)*(np.power(1.+255, np.abs(x)) - 1.)

def audio_signal_to_classes(audio):
    audio=np.floor(128*mu_law(audio))
    audio = np.clip(audio, -128., 128.) / 128.
    audio = (audio + 1.) / 2.
    audio = np.round(audio * (FLAGS.num_audio_classes - 1)).astype(np.int)
    return audio

def audio_classes_to_signal_th(audio):
    audio = audio.detach().cpu().numpy()
    audio = audio.astype(np.float32) / (FLAGS.num_audio_classes - 1)
    audio = audio * 2. - 1.
    audio = audio * 128.
    audio = mu_law_inverse(audio / 128.)
    audio = torch.from_numpy(audio)
    return audio

class GlobalMelSpecDiscretizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #IPython.embed()
        if FLAGS.patient_eight:
            self.centroids=torch.load('global_centroids_kh8.pt')
        elif FLAGS.patient_thirteen:
            raise ValueError('Patient 13 has no global centroids')
        else:
            self.centroids=torch.load('global_centroids_kh4.pt')


    def mel_to_class(self,melspecs):
        centroids=self.centroids.repeat(melspecs.shape[0]).view(-1,FLAGS.num_mel_centroids).cuda()
        distances=torch.abs((melspecs.unsqueeze(1)-centroids.unsqueeze(-1).unsqueeze(-1))).transpose(1,2)  # bs x seq_len x 12 x mel_bins
        cluster_assignments=torch.argmin(distances,dim=2) #closest enter of each entry # bs x seq_len x mel_bins
        
        return cluster_assignments
    def class_to_centroids(self,cluster_assignments):
        values=torch.zeros_like(cluster_assignments).float()
        for i in range(len(self.centroids)):
            values[cluster_assignments==i]=self.centroids[i].cuda()
        return values

    def mel_to_centroids(self,melspecs): 
        cluster_assignments=self.mel_to_class(melspecs)
        return self.class_to_centroids(cluster_assignments)
       
class LocalMelSpecDiscretizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if FLAGS.patient_eight:
            self.centroids=torch.load("local_centroids_kh8.pt").cuda()
        elif FLAGS.patient_thirteen:
            raise ValueError('Patient 13 has no local centroids')
        else:
            self.centroids=torch.load("local_centroids_kh4.pt").cuda()

    def mel_to_class(self,melspecs):
        #compute distances
        melspecs=melspecs.transpose(1,2) # from [bs x seq x bins] to [bs x bins x seq]

        distances=torch.abs((melspecs.transpose(0,1).unsqueeze(1)-self.centroids.unsqueeze(-1).unsqueeze(-1)))
        cluster_assignments=torch.argmin(distances,dim=1) #[bins x bs x seq_len]
        
        return cluster_assignments.transpose(0,1).transpose(1,2) # [bs x seq_len x bins]
    def class_to_centroids(self,cluster_assignments):
        cluster_assignments=cluster_assignments.transpose(1,2) # from [bs x seq_len x bins] to [bs x bins x seq]
        cluster_assignments=cluster_assignments.transpose(0,1) # [bins x bs x seq]
        values=torch.zeros_like(cluster_assignments).float() # [bins x bs x seq]
        for i in range(len(self.centroids)):
            for j in range(self.centroids.shape[1]):
                indices_to_replace=(cluster_assignments==j)[i] #bs x seq_len
                values[i,indices_to_replace]=self.centroids[i,j]

        return values.transpose(0,1).transpose(1,2) #[bs x seq_len x bins ]

    def mel_to_centroids(self,melspecs): 
        cluster_assignments=self.mel_to_class(melspecs)
        return self.class_to_centroids(cluster_assignments)
       
class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, #50ms*20480Hz=1024,12.5ms*20480Hz=256 
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0): 
    # def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
    #              n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
    #              mel_fmax=8000.0): 
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length) # hop and window length are in samples.
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)    ### filter_length = number of FFT components

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        if (FLAGS.OLS or FLAGS.DenseModel):
            return mel_output[:,:,3].unsqueeze(-1)
            #stft_fn.transform pads sequence with reflection to be twice the original size.
            #hence 5 MFCC framea are produced for the 50ms window. We take the middle one which should correspond best to the original frame.
        else:
            return mel_output

def create_audio_plot(audios_with_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for audio, label in audios_with_labels:
        ax.plot(audio.detach().cpu().numpy(), label=label)
    ax.legend(loc='best')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return torch.from_numpy(data).float() / 255

def create_MFCC_plot(MFCCs, targets):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].imshow(targets.T.detach().cpu().numpy(), cmap='viridis',aspect='auto') 
    
    ax[1].imshow(MFCCs.T.detach().cpu().numpy(), cmap='viridis',aspect='auto') 
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # return torch.from_numpy(data.transpose(1,0,2)).float() / 255
    return torch.from_numpy(data).float() / 255

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG(data, sr):
    """
    Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    #numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Create feature space
    data = np.abs(hilbert3(data))
    # feat = np.zeros((numWindows,data.shape[1]))
    # for win in range(numWindows):
    #     start= int(np.floor((win*frameshift)*sr))
    #     stop = int(np.floor(start+windowLength*sr))
    #     feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return data

# def k_fold_split(eeg, audio, k=10, current_fold=0, split='train', hop=None):
#     """
#     进行 k 折交叉验证,其中验证集占1份,测试集占1份,剩余为训练集
    
#     参数:
#     eeg -- EEG 数据
#     audio -- 音频数据
#     k -- 交叉验证的折数
#     current_fold -- 当前正在处理的折数 (从 0 开始)
#     """
#     audio_eeg_sample_ratio = len(audio) / len(eeg) 

#     # 计算每一折的样本数
#     total_samples_eeg = len(eeg)
#     fold_size_eeg = total_samples_eeg // k
#     val_size_eeg = fold_size_eeg
#     test_size_eeg = fold_size_eeg
#     train_size_eeg = total_samples_eeg - val_size_eeg - test_size_eeg
    
#     total_samples_audio = len(audio)
#     fold_size_audio = total_samples_audio // k
#     val_size_audio = fold_size_audio
#     test_size_audio = fold_size_audio
#     train_size_audio = total_samples_audio - val_size_audio - test_size_audio
    
#     # 计算训练集、验证集和测试集的起始和结束索引
#     train2_start_eeg = 0
#     train2_end_eeg = 0
    
#     train2_start_audio = 0
#     train2_end_audio = 0
#     if current_fold == 0:
#         val_start_eeg = (k-1) * fold_size_eeg
#         val_end_eeg = total_samples_eeg
        
#         test_start_eeg = 0
#         test_end_eeg = test_size_eeg

#         train1_start_eeg = test_end_eeg
#         train1_end_eeg = val_start_eeg
        
#         val_start_audio = (k-1) * fold_size_audio
#         val_end_audio = total_samples_audio
        
#         test_start_audio = 0
#         test_end_audio = test_size_audio

#         train1_start_audio = test_end_audio
#         train1_end_audio = val_start_audio
#     elif current_fold == 1:
#         val_start_eeg = 0
#         val_end_eeg = val_size_eeg
        
#         test_start_eeg = val_end_eeg
#         test_end_eeg = test_start_eeg + test_size_eeg

#         train1_start_eeg = test_end_eeg
#         train1_end_eeg = total_samples_eeg
        
#         val_start_audio = 0
#         val_end_audio = val_size_audio
        
#         test_start_audio = val_end_audio
#         test_end_audio = test_start_audio + test_size_audio

#         train1_start_audio = test_end_audio
#         train1_end_audio = total_samples_audio
#     elif current_fold < k-1:
#         val_start_eeg = (current_fold-1) * fold_size_eeg
#         val_end_eeg = val_start_eeg + val_size_eeg
        
#         test_start_eeg = val_end_eeg
#         test_end_eeg = test_start_eeg + test_size_eeg

#         train1_start_eeg = 0
#         train1_end_eeg = val_start_eeg
#         train2_start_eeg = test_end_eeg
#         train2_end_eeg = total_samples_eeg
        
#         val_start_audio = (current_fold-1) * fold_size_audio
#         val_end_audio = val_start_audio + val_size_audio
        
#         test_start_audio = val_end_audio
#         test_end_audio = test_start_audio + test_size_audio

#         train1_start_audio = 0
#         train1_end_audio = val_start_audio
#         train2_start_audio = test_end_audio
#         train2_end_audio = total_samples_audio
#     elif current_fold == k-1:
#         val_start_eeg = (current_fold-1) * fold_size_eeg
#         val_end_eeg = val_start_eeg + val_size_eeg
        
#         test_start_eeg = val_end_eeg
#         test_end_eeg = total_samples_eeg

#         train1_start_eeg = 0
#         train1_end_eeg = val_start_eeg
        
#         val_start_audio = (current_fold-1) * fold_size_audio
#         val_end_audio = val_start_audio + val_size_audio
        
#         test_start_audio = val_end_audio
#         test_end_audio = total_samples_audio

#         train1_start_audio = 0
#         train1_end_audio = val_start_audio
    
#     # 划分数据集
#     if split == 'train':
#         if train2_start_eeg == 0 and train2_end_eeg ==0:
#             print(f'train1 eeg:{train1_start_eeg}~{train1_end_eeg}')
#             print(f'train1 audio:{train1_start_audio}~{train1_end_audio}')
#             print(f'train2 eeg:{train2_start_eeg}~{train2_end_eeg}')
#             print(f'train2 audio:{train2_start_audio}~{train2_end_audio}')
#             eeg1 = eeg[train1_start_eeg:train1_end_eeg]
#             audio1 = audio[train1_start_audio:train1_end_audio]
            
#             eeg1 = torch.from_numpy(eeg1).float()
#             audio1 = torch.from_numpy(audio1).float()
#             dataset = EEGAudioDataset(eeg1, audio1, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
#         else:
#             print(f'train1 eeg:{train1_start_eeg}~{train1_end_eeg}')
#             print(f'train1 audio:{train1_start_audio}~{train1_end_audio}')
#             print(f'train2 eeg:{train2_start_eeg}~{train2_end_eeg}')
#             print(f'train2 audio:{train2_start_audio}~{train2_end_audio}')
#             eeg1 = eeg[train1_start_eeg:train1_end_eeg]
#             eeg2 = eeg[train2_start_eeg:train2_end_eeg]
#             audio1 = audio[train1_start_audio:train1_end_audio]
#             audio2 = audio[train2_start_audio:train2_end_audio]
            
#             eeg1 = torch.from_numpy(eeg1).float()
#             eeg2 = torch.from_numpy(eeg2).float()
#             audio1 = torch.from_numpy(audio1).float()
#             audio2 = torch.from_numpy(audio2).float()

#             dataset1 = EEGAudioDataset(eeg1, audio1, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
#             dataset2 = EEGAudioDataset(eeg2, audio2, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
#             dataset = ConcatDataset([dataset1, dataset2])
#     elif split == 'val':
#         print(f'val eeg:{val_start_eeg}~{val_end_eeg}')
#         print(f'val audio:{val_start_audio}~{val_end_audio}')
#         eeg = eeg[val_start_eeg:val_end_eeg]
#         audio = audio[val_start_audio:val_end_audio]

#         eeg = torch.from_numpy(eeg).float()
#         audio = torch.from_numpy(audio).float()
        
#         dataset = EEGAudioDataset(eeg, audio, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
    
#     elif split == 'test':
#         print(f'test eeg:{test_start_eeg}~{test_end_eeg}')
#         print(f'test audio:{test_start_audio}~{test_end_audio}')
#         eeg = eeg[test_start_eeg:test_end_eeg]
#         audio = audio[test_start_audio:test_end_audio]
    
#         eeg = torch.from_numpy(eeg).float()
#         audio = torch.from_numpy(audio).float()

#         dataset = EEGAudioDataset(eeg, audio, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
#     return dataset


def k_fold_split(eeg, audio, k=10, current_fold=0, split='train', hop=None):
    """
    进行 k 折交叉验证,其中验证集占1份,测试集占1份,剩余为训练集
    
    参数:
    eeg -- EEG 数据
    audio -- 音频数据
    k -- 交叉验证的折数
    current_fold -- 当前正在处理的折数 (从 0 开始)
    """
    audio_eeg_sample_ratio = len(audio) / len(eeg) 

    # 计算每一折的样本数
    total_samples_eeg = len(eeg)
    fold_size_eeg = total_samples_eeg // k
    test_size_eeg = fold_size_eeg
    train_size_eeg = total_samples_eeg - test_size_eeg
    
    total_samples_audio = len(audio)
    fold_size_audio = total_samples_audio // k
    test_size_audio = fold_size_audio
    train_size_audio = total_samples_audio - test_size_audio
    
    # 计算训练集、验证集和测试集的起始和结束索引
    train2_start_eeg = 0
    train2_end_eeg = 0
    
    train2_start_audio = 0
    train2_end_audio = 0
    if current_fold == 0:
        test_start_eeg = 0
        test_end_eeg = test_size_eeg

        train1_start_eeg = test_end_eeg
        train1_end_eeg = total_samples_eeg
        
        test_start_audio = 0
        test_end_audio = test_size_audio

        train1_start_audio = test_end_audio
        train1_end_audio = total_samples_audio
    elif current_fold < k-1:
        
        test_start_eeg = current_fold * fold_size_eeg
        test_end_eeg = test_start_eeg + test_size_eeg

        train1_start_eeg = 0
        train1_end_eeg = test_start_eeg
        train2_start_eeg = test_end_eeg
        train2_end_eeg = total_samples_eeg
        
        test_start_audio = current_fold * fold_size_audio
        test_end_audio = test_start_audio + test_size_audio

        train1_start_audio = 0
        train1_end_audio = test_start_audio
        train2_start_audio = test_end_audio
        train2_end_audio = total_samples_audio
    elif current_fold == k-1:
        
        test_start_eeg = current_fold * fold_size_eeg
        test_end_eeg = total_samples_eeg

        train1_start_eeg = 0
        train1_end_eeg = test_start_eeg
        
        test_start_audio = current_fold * fold_size_audio
        test_end_audio = total_samples_audio

        train1_start_audio = 0
        train1_end_audio = test_start_audio
    
    # 划分数据集
    if split == 'train':
        if train2_start_eeg == 0 and train2_end_eeg ==0:
            print(f'train1 eeg:{train1_start_eeg}~{train1_end_eeg}')
            print(f'train1 audio:{train1_start_audio}~{train1_end_audio}')
            print(f'train2 eeg:{train2_start_eeg}~{train2_end_eeg}')
            print(f'train2 audio:{train2_start_audio}~{train2_end_audio}')
            eeg1 = eeg[train1_start_eeg:train1_end_eeg]
            audio1 = audio[train1_start_audio:train1_end_audio]
            
            eeg1 = torch.from_numpy(eeg1).float()
            audio1 = torch.from_numpy(audio1).float()
            dataset = EEGAudioDataset(eeg1, audio1, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
        else:
            print(f'train1 eeg:{train1_start_eeg}~{train1_end_eeg}')
            print(f'train1 audio:{train1_start_audio}~{train1_end_audio}')
            print(f'train2 eeg:{train2_start_eeg}~{train2_end_eeg}')
            print(f'train2 audio:{train2_start_audio}~{train2_end_audio}')
            eeg1 = eeg[train1_start_eeg:train1_end_eeg]
            eeg2 = eeg[train2_start_eeg:train2_end_eeg]
            audio1 = audio[train1_start_audio:train1_end_audio]
            audio2 = audio[train2_start_audio:train2_end_audio]
            
            eeg1 = torch.from_numpy(eeg1).float()
            eeg2 = torch.from_numpy(eeg2).float()
            audio1 = torch.from_numpy(audio1).float()
            audio2 = torch.from_numpy(audio2).float()

            dataset1 = EEGAudioDataset(eeg1, audio1, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
            dataset2 = EEGAudioDataset(eeg2, audio2, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
            dataset = ConcatDataset([dataset1, dataset2])
    elif split == 'test':
        print(f'test eeg:{test_start_eeg}~{test_end_eeg}')
        print(f'test audio:{test_start_audio}~{test_end_audio}')
        eeg = eeg[test_start_eeg:test_end_eeg]
        audio = audio[test_start_audio:test_end_audio]
    
        eeg = torch.from_numpy(eeg).float()
        audio = torch.from_numpy(audio).float()

        dataset = EEGAudioDataset(eeg, audio, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
    return dataset
    
    

def get_data(k=5, current_fold=0, split='train', hop=None):
    if FLAGS.dataset == 'word':
        data_dir = Path('/hdd/hyy/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS') 
    elif FLAGS.dataset == 'sentence':
        data_dir = Path('/hdd/hyy/SentenceProductionDutch') 

    if FLAGS.dataset == 'word':
        #load data
        io = NWBHDF5IO(os.path.join(data_dir,FLAGS.patient,'ieeg',f'{FLAGS.patient}_task-wordProduction_ieeg.nwb'), 'r')
        nwbfile = io.read()
        #load seeg
        eeg = nwbfile.acquisition['iEEG'].data[:]
        FLAGS.sampling_rate_eeg = 1024
        eeg = extractHG(eeg,FLAGS.sampling_rate_eeg)
        #load audio
        audio = nwbfile.acquisition['Audio'].data[:]
        audioSamplingRate = 48000
        targetSR = 22050
        audio=resample(audio, orig_sr=audioSamplingRate, target_sr=targetSR)
        #16位int 类型 取值范围 -32768~32767
        scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)  #这里需要确定一下
        io.close()
    elif FLAGS.dataset == 'sentence':
        # if FLAGS.patient_eight:
        #     eeg=np.load(str(data_dir / "p1_sEEG.npy"))
        #     audio=np.load(str(data_dir / "p1_audio_final.npy"))
        # elif FLAGS.patient_thirteen:
        #     eeg=np.load(str(data_dir / "p2_sEEG.npy"))
        #     audio=np.load(str(data_dir / "p2_audio_final.npy"))
        #     audio[np.where(audio>1)]=0.9999 #some outliers
        #     audio[np.where(audio<-1)]=-0.9999
        # else:    
        #     eeg=np.load(str(data_dir / "p3_sEEG.npy"))
        #     audio=np.load(str(data_dir / "p3_audio_final.npy"))
        eeg=np.load(str(data_dir / f'{FLAGS.patient}_sEEG.npy'))
        audio=np.load(str(data_dir / f'{FLAGS.patient}_audio_final.npy'))
        FLAGS.sampling_rate_eeg = 1024
        eeg = extractHG(eeg,FLAGS.sampling_rate_eeg)
        audioSamplingRate = 22050
        targetSR = 22050
        #audio=resample(audio, orig_sr=audioSamplingRate, target_sr=targetSR)

    # audio_eeg_sample_ratio = len(audio) / len(eeg) #make this an int??
    
    # if not FLAGS.use_MFCCs:
    #     audio = audio_signal_to_classes(audio)
    
    # num_train_samples = round(len(eeg) * FLAGS.train_split)
    # num_train_samples_audio = round(len(audio) * FLAGS.train_split)
    
    # num_val_samples = round(len(eeg) * FLAGS.val_split)
    # num_val_samples_audio = round(len(audio) * FLAGS.val_split)
    
    # num_test_samples = len(eeg)-num_train_samples-num_val_samples
    # num_test_samples_audio = len(audio)-num_train_samples_audio-num_val_samples_audio

    # test_set_beginning=False
    # if test_set_beginning:
    #     if split == 'train':
    #         eeg = eeg[num_test_samples:]
    #         audio = audio[num_test_samples_audio:]
    #     elif split == 'test':
    #         eeg = eeg[:num_test_samples]
    #         audio = audio[:num_test_samples_audio]
    # else:
    #     if split == 'train':
    #         eeg = eeg[:num_train_samples]
    #         audio = audio[:num_train_samples_audio]
    #     elif split == 'val':
    #         eeg = eeg[num_train_samples:num_train_samples+num_val_samples]
    #         audio = audio[num_train_samples_audio:num_train_samples_audio+num_val_samples_audio]
    #     elif split == 'test':
    #         eeg = eeg[num_train_samples+num_val_samples:]
    #         audio = audio[num_train_samples_audio+num_val_samples_audio:]

    # if FLAGS.double_trouble and split == 'train':
    #     np.random.shuffle(audio)
    #     np.random.shuffle(eeg)

    # eeg = torch.from_numpy(eeg).float()

    # if FLAGS.use_MFCCs:
    #     audio = torch.from_numpy(audio).float()
    # else:
    #     audio = torch.from_numpy(audio).long()

    # return EEGAudioDataset(eeg, audio, FLAGS.num_audio_classes,audio_eeg_sample_ratio, hop=hop)
    
    dataset = k_fold_split(eeg, audio, k, current_fold, split, hop)
    return dataset
    



class EEGAudioDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, audio, num_audio_classes,audio_eeg_sample_ratio, hop=None):
        super().__init__()

        self.audio_eeg_sample_ratio=audio_eeg_sample_ratio
        self.sampling_rate_audio=FLAGS.sampling_rate_eeg*self.audio_eeg_sample_ratio

        self.eeg = eeg
        self.audio = audio

        self.num_audio_classes = num_audio_classes #only meaningful if direct audio is synthesized

        window_size_eeg=FLAGS.window_size / 1000 * FLAGS.sampling_rate_eeg
        self.window_size_eeg = round(window_size_eeg)
        self.versatz_eeg=round(window_size_eeg*FLAGS.versatz_windows)
        self.window_size_audio=round(window_size_eeg * self.audio_eeg_sample_ratio)

        self.hop =  self.window_size_eeg if hop is None else int(hop/1000*FLAGS.sampling_rate_eeg)
        self.tacotron_mel_transformer=TacotronSTFT(n_mel_channels=FLAGS.num_mel) #all default values are used, i.e. ~ 50ms window size, 12.5 ms hop, 80 mel bins, 8000hz max frequency.
        logging.info(f'''
        num_audio_classes: {self.num_audio_classes},
        window_size_eeg: {self.window_size_eeg}, 
        versatz_eeg: {self.versatz_eeg}, 
        window_size_audio: {self.window_size_audio},
        hop: {self.hop}
        ''')

    def __len__(self):
        num_samples = len(self.eeg) - 2 * self.versatz_eeg - self.window_size_eeg
        return num_samples // self.hop # integer division

    def __getitem__(self, idx):
        idx_eeg = idx*self.hop+self.versatz_eeg
        idx_audio = round(idx_eeg*self.audio_eeg_sample_ratio) #(includes versatz in audio)
        eeg = self.eeg[idx_eeg-self.versatz_eeg:idx_eeg+self.window_size_eeg+self.versatz_eeg]
        audio = self.audio[idx_audio:idx_audio+self.window_size_audio]
        return eeg, audio
