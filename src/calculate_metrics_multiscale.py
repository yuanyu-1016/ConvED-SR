import os
import torch
import numpy as np
import pickle
import pandas as pd
import soundfile as sf
import librosa
from pymcd.mcd import Calculate_MCD
from pystoi import stoi as Calculate_STOI
import webrtcvad
import parselmouth
import scipy




def MCD(orig_audio_path, rec_audio_path):
    # three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcd = mcd_toolbox.calculate_mcd(orig_audio_path, rec_audio_path)
    return mcd
    
def STOI(orig_audio, rec_audio, sr):
    stoi = Calculate_STOI(orig_audio, rec_audio, sr, extended=False)
    return stoi

def eSTOI(orig_audio, rec_audio, sr):
    estoi = Calculate_STOI(orig_audio, rec_audio, sr, extended=True)
    return estoi

def PCC(orig_mel, rec_mel):
    # pearson_r=torch.nn.functional.cosine_similarity(rec_mel-torch.mean(rec_mel,dim=1).unsqueeze(1),orig_mel-torch.mean(orig_mel,dim=1).unsqueeze(1),dim=1)   
    pearson_r=np.mean([np.corrcoef(orig_mel[:, i], rec_mel[:, i])[0, 1] for i in range(orig_mel.shape[1])])
    return pearson_r

def MSE(orig_mel, rec_mel):
    # criterion = torch.nn.MSELoss(reduction='none')
    # mse = criterion(orig_mel, rec_mel)#[bs * seq_len]
    mse = np.mean(np.mean(np.square(orig_mel - rec_mel), axis=1))
    return mse

def get_vad(audio, sr, frame_duration_ms=30):
    """
    使用 webrtcvad 库从原始语音信号中提取 VAD 标签。
    
    参数:
    audio (np.ndarray): 输入的原始语音信号, 形状为 (N,)
    sample_rate (int): 采样率, 单位为 Hz
    frame_duration_ms (int): 帧长, 单位为毫秒
    
    返回:
    np.ndarray: 语音活动检测标签, 形状为 (T,), 其中 T 为总帧数
    """
    # 初始化 webrtcvad 检测器
    vad = webrtcvad.Vad(mode=3)

    #webrtcvad 支持 8kHz、16kHz 和 32kHz
    audio_downsampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000
    # 计算每帧的长度
    frame_length = int(sr * (frame_duration_ms / 1000))
    
    # 遍历每一帧并进行 VAD 检测
    vad_labels = []
    for i in range(0, len(audio-frame_length), frame_length):
        frame = audio[i:i+frame_length]
        is_speech = vad.is_speech(frame.tobytes(), sr)
        vad_labels.append(1 if is_speech else 0)
    
    return np.array(vad_labels)

def VAD_match(orig_audio, rec_audio,sr):
    """
    计算两个语音信号的 VAD 匹配程度。
    
    参数:
    vad1 (np.ndarray): 第一个语音信号的 VAD 标签, 形状为 (T,)
    vad2 (np.ndarray): 第二个语音信号的 VAD 标签, 形状为 (T,)
    
    返回:
    float: VAD 匹配度, 范围为 [0, 1]
    """
    vad1 = get_vad(orig_audio, sr)
    vad2 = get_vad(rec_audio, sr)
    
    # 计算 VAD 匹配的时间步数
    match_count = np.sum(vad1 == vad2)
    
    # 计算 VAD 匹配度
    vad_match_score = match_count / len(vad1)
    
    return vad_match_score

def Pitch_match(orig_audio, rec_audio, sr):
    """
    计算两段语音的 pitch 匹配度。
    
    参数:
    audio1 (np.ndarray): 第一段语音信号, 形状为 (N,)
    audio2 (np.ndarray): 第二段语音信号, 形状为 (M,)
    sample_rate (int): 采样率, 单位为 Hz
    
    返回:
    float: 两段语音的 pitch 匹配度, 范围为 [0, 1]
    """
    # 使用 parselmouth 计算两段语音的 pitch 轨迹
    sound1 = parselmouth.Sound(orig_audio, sr)
    sound2 = parselmouth.Sound(rec_audio, sr)
    pitch1 = sound1.to_pitch()
    pitch2 = sound2.to_pitch()
    
    # 获取 pitch 轨迹
    pitch_values1 = pitch1.selected_array['frequency']
    pitch_values2 = pitch2.selected_array['frequency']
    
    # 对齐 pitch 轨迹的长度
    min_length = min(len(pitch_values1), len(pitch_values2))
    pitch_values1 = pitch_values1[:min_length]
    pitch_values2 = pitch_values2[:min_length]
    
    # 计算 pitch 匹配度
    pitch_diff = np.abs(pitch_values1 - pitch_values2)
    pitch_match_score =  1 - np.mean(pitch_diff) / np.max([pitch_values1.mean(), pitch_values2.mean()])
    
    return pitch_match_score

def downsample(data, method='slice', factor=2):
    """
    对输入数据进行下采样。

    参数:
    - data: 输入的二维数组 (T, N)
    - method: 下采样方法 ('slice', 'average_pool', 'max_pool', 'resample')
    - factor: 下采样因子 (默认为2)

    返回:
    - 下采样后的数组
    """
    if method == 'slice':
        # 切片下采样
        return data[::factor, ::factor]
    
    elif method == 'avg_pool':
        # 平均池化下采样
        # 计算新的维度
        new_T = (data.shape[0] + factor - 1) // factor  # 向上取整
        new_N = (data.shape[1] + factor - 1) // factor  # 向上取整

        # 填充数据以适应reshape
        padded_data = np.pad(data, ((0, new_T * factor - data.shape[0]), (0, new_N * factor - data.shape[1])), mode='constant')

        # 重塑并计算平均
        reshaped = padded_data.reshape(new_T, factor, new_N, factor)
        return reshaped.mean(axis=(1, 3))
        
    elif method == 'max_pool':
        # 最大池化下采样
        # 计算新的维度
        new_T = (data.shape[0] + factor - 1) // factor  # 向上取整
        new_N = (data.shape[1] + factor - 1) // factor  # 向上取整

        # 填充数据以适应reshape
        padded_data = np.pad(data, ((0, new_T * factor - data.shape[0]), (0, new_N * factor - data.shape[1])), mode='constant')

        # 重塑并计算平均
        reshaped = padded_data.reshape(new_T, factor, new_N, factor)
        return reshaped.max(axis=(1, 3))
    
    elif method == 'scipy':
        # 使用SciPy的resample函数
        new_length = data.shape[0] // factor
        return scipy.signal.resample(data, new_length, axis=0)

    # elif method == 'interpolate':
    #     # 插值下采样
    #     x = np.arange(data.shape[0])
    #     f = interp1d(x, data, axis=0, kind='linear', fill_value='extrapolate')
    #     x_new = np.linspace(0, data.shape[0] - 1, num=(data.shape[0] + factor - 1) // factor)
    #     return f(x_new)
    
    else:
        raise ValueError("Unsupported downsampling method: choose from 'slice', 'avg_pool', 'max_pool', 'scipy'.")

def calculate_audio_metrics(orig_audio_path, rec_audio_path):
    orig_audio, sr = sf.read(orig_audio_path)
    rec_audio, sr = sf.read(rec_audio_path)
   
    mcd = MCD(orig_audio_path , rec_audio_path)
    stoi = STOI(orig_audio, rec_audio, sr)
    estoi = eSTOI(orig_audio, rec_audio, sr)
    # vad_match = 0
    # pitch_match = 0
    # vad_match = VAD_match(orig_audio, rec_audio, sr)
    # pitch_match = Pitch_match(orig_audio, rec_audio, sr)
    # return {'mcd':mcd, 'stoi':stoi, 'estoi':estoi, 'vad_match':vad_match, 'pitch_match':pitch_match}
    return {'mcd':mcd, 'stoi':stoi, 'estoi':estoi}
    
def calculate_mel_metrics(orig_mel_path, rec_mel_path):
    downsample_method = 'scipy'
    orig_mel = np.load(orig_mel_path)
    rec_mel = np.load(rec_mel_path)
    pcc_1 = PCC(orig_mel, rec_mel)
    mse_1 = MSE(orig_mel, rec_mel)

    # 对 orig_mel 和 rec_mel 两倍下采样
    orig_mel = downsample(orig_mel, method=downsample_method, factor=2)
    rec_mel = downsample(rec_mel, method=downsample_method, factor=2)
    pcc_2 = PCC(orig_mel, rec_mel)
    mse_2 = MSE(orig_mel, rec_mel)

    # 对 orig_mel 和 rec_mel 两倍下采样
    orig_mel = downsample(orig_mel, method=downsample_method, factor=2)
    rec_mel = downsample(rec_mel, method=downsample_method, factor=2)
    pcc_3 = PCC(orig_mel, rec_mel)
    mse_3 = MSE(orig_mel, rec_mel)
    
    # 对 orig_mel 和 rec_mel 两倍下采样
    orig_mel = downsample(orig_mel, method=downsample_method, factor=2)
    rec_mel = downsample(rec_mel, method=downsample_method, factor=2)
    pcc_4 = PCC(orig_mel, rec_mel)
    mse_4 = MSE(orig_mel, rec_mel)
    
    # 对 orig_mel 和 rec_mel 两倍下采样
    orig_mel = downsample(orig_mel, method=downsample_method, factor=2)
    rec_mel = downsample(rec_mel, method=downsample_method, factor=2)
    pcc_5 = PCC(orig_mel, rec_mel)
    mse_5 = MSE(orig_mel, rec_mel)

    # 对 orig_mel 和 rec_mel 两倍下采样
    orig_mel = downsample(orig_mel, method=downsample_method, factor=2)
    rec_mel = downsample(rec_mel, method=downsample_method, factor=2)
    pcc_6 = PCC(orig_mel, rec_mel)
    mse_6 = MSE(orig_mel, rec_mel)

    # # 对 orig_mel 和 rec_mel 两倍下采样
    # orig_mel = downsample(orig_mel, method=downsample_method, factor=2)
    # rec_mel = downsample(rec_mel, method=downsample_method, factor=2)
    # pcc_7 = PCC(orig_mel, rec_mel)
    # mse_7 = MSE(orig_mel, rec_mel)
    # return {'pcc_1':pcc_1,'pcc_2':pcc_2,'pcc_3':pcc_3,'pcc_4':pcc_4,'pcc_5':pcc_5,'mse_1':mse_1,'mse_2':mse_2,'mse_3':mse_3,'mse_4':mse_4,'mse_5':mse_5}
    return {'pcc_1':pcc_1,'pcc_2':pcc_2,'pcc_3':pcc_3,'pcc_4':pcc_4,'pcc_5':pcc_5,'pcc_6':pcc_6,'mse_1':mse_1,'mse_2':mse_2,'mse_3':mse_3,'mse_4':mse_4,'mse_5':mse_5,'mse_6':mse_6}

def print_metrics(metrics):
    for key, value in metrics.items():
        print(f'{key.upper()}: {value:.3f}')
    print('')

# def test_metrics(dataset, patient, fold=None, orig_result_path=None):
#     # result_path = f'results/{dataset}/{patient}'
#     if fold is not None:
#         result_path = os.path.join(orig_result_path, f'{dataset}/{patient}/fold_{fold}')
#         print(f'Calculate metrics for {dataset} dataset, patient {patient}, fold {fold}')
#     else:
#         result_path = os.path.join(orig_result_path, f'{dataset}/{patient}')
#         print(f'Calculate metrics for {dataset} dataset, patient {patient}')

#     metrics_audio_real_pred = calculate_audio_metrics(os.path.join(result_path, 'audio_real.wav'), os.path.join(result_path, 'audio_pred.wav'))
#     metrics_mel_real_pred = calculate_mel_metrics(os.path.join(result_path, 'mel_spec_real.npy'), os.path.join(result_path, 'mel_spec_pred.npy'))
    
#     metrics = {
#     'mcd': metrics_audio_real_pred['mcd'],
#     'stoi': metrics_audio_real_pred['stoi'], 
#     'estoi': metrics_audio_real_pred['estoi'],
#     'pcc': metrics_mel_real_pred['pcc'],
#     'mse': metrics_mel_real_pred['mse']
#     }


#     #save metrics
#     # with open(os.path.join(result_path, 'metrics.pkl'), 'wb') as f:
#     #     pickle.dump(metrics,f)
#     #     f.close()

#     metrics_excel = pd.DataFrame.from_dict(metrics, orient='index')
#     print(metrics_excel)

#     return metrics

def calculate_metrics(dataset, patient, fold=None, orig_result_path=None):
    # result_path = f'results/{dataset}/{patient}'
    if fold is not None:
        result_path = os.path.join(orig_result_path, f'{dataset}/{patient}/fold_{fold}')
        print(f'Calculate metrics for {dataset} dataset, patient {patient}, fold {fold}')
    else:
        result_path = os.path.join(orig_result_path, f'{dataset}/{patient}')
        print(f'Calculate metrics for {dataset} dataset, patient {patient}')

    # metrics_audio_real_rec = calculate_audio_metrics(os.path.join(result_path, 'audio_real.wav'), os.path.join(result_path, 'audio_rec.wav')) #base metrics
    # metrics_audio_real_pred = calculate_audio_metrics(os.path.join(result_path, 'audio_real.wav'), os.path.join(result_path, 'audio_pred.wav'))
    # metrics_audio_rec_pred = calculate_audio_metrics(os.path.join(result_path, 'audio_rec.wav'), os.path.join(result_path, 'audio_pred.wav'))
    # print('metrics_audio_real_rec:')
    # print_metrics(metrics_audio_real_rec)
    # print('metrics_audio_real_pred:')
    # print_metrics(metrics_audio_real_pred)
    # print('metrics_audio_rec_pred:')
    # print_metrics(metrics_audio_rec_pred)

    metrics_mel_real_pred = calculate_mel_metrics(os.path.join(result_path, 'mel_spec_real.npy'), os.path.join(result_path, 'mel_spec_pred.npy'))
    # print('metrics_mel_real_pred:')
    # print_metrics(metrics_mel_real_pred)

    metrics = {
        # 'audio_real_rec': metrics_audio_real_rec,
        # 'audio_real_pred': metrics_audio_real_pred,
        # 'audio_rec_pred': metrics_audio_rec_pred,
        'mel_real_pred': metrics_mel_real_pred
    }


    #save metrics
    with open(os.path.join(result_path, 'metrics_multiscale.pkl'), 'wb') as f:
        pickle.dump(metrics,f)
        f.close()

    # 字典结果写入excel表保存
    # metrics_excel = pd.DataFrame(metrics)
    # metrics_excel = metrics_excel.T
    # print(metrics_excel)
    # metrics_excel.to_excel(os.path.join(metrics_dir, 'metrics.xlsx'))

    # metrics_excel = pd.DataFrame.from_dict(metrics, orient='index')
    # print(metrics_excel)
    # metrics_excel.to_excel(os.path.join(result_path, 'metrics.xlsx'), index=True)

    return metrics

def save_singlesub_metrics_to_excel(dataset, patient, orig_result_path):
    # if dataset == 'sentence':
    #     pt = 'p%d'%(patient+1)
    # elif dataset == 'word':
    #     pt = 'sub-%02d'%(patient+1)
    metrics_all = {}
    # metrics_mean = {'pcc_1': [], 'pcc_2': [], 'pcc_3': [], 'pcc_4': [], 'pcc_5': [], 'mse_1': [], 'mse_2': [], 'mse_3': [], 'mse_4': [], 'mse_5': []}
    # metrics_std = {'pcc_1': [], 'pcc_2': [], 'pcc_3': [], 'pcc_4': [], 'pcc_5': [], 'mse_1': [], 'mse_2': [], 'mse_3': [], 'mse_4': [], 'mse_5': []}
    
    metrics_mean = {'pcc_1': [], 'pcc_2': [], 'pcc_3': [], 'pcc_4': [], 'pcc_5': [], 'pcc_6': [], 'mse_1': [], 'mse_2': [], 'mse_3': [], 'mse_4': [], 'mse_5': [], 'mse_6': []}
    metrics_std = {'pcc_1': [], 'pcc_2': [], 'pcc_3': [], 'pcc_4': [], 'pcc_5': [], 'pcc_6': [], 'mse_1': [], 'mse_2': [], 'mse_3': [], 'mse_4': [], 'mse_5': [], 'mse_6': []}
    for fold in range(10):
        # result_path = f'results/{dataset}/{patient}'
        result_path = os.path.join(orig_result_path, f'{dataset}/{patient}/fold_{fold}')
        with open(os.path.join(result_path, 'metrics_multiscale.pkl'), 'rb') as f:
            metrics = pickle.load(f)
            f.close()

        # metrics_save = {
        # 'pcc_1': metrics['mel_real_pred']['pcc_1'],
        # 'pcc_2': metrics['mel_real_pred']['pcc_2'],
        # 'pcc_3': metrics['mel_real_pred']['pcc_3'],
        # 'pcc_4': metrics['mel_real_pred']['pcc_4'],
        # 'pcc_5': metrics['mel_real_pred']['pcc_5'],
        # 'mse_1': metrics['mel_real_pred']['mse_1'],
        # 'mse_2': metrics['mel_real_pred']['mse_2'],
        # 'mse_3': metrics['mel_real_pred']['mse_3'],
        # 'mse_4': metrics['mel_real_pred']['mse_4'],
        # 'mse_5': metrics['mel_real_pred']['mse_5'],
        # }
        metrics_save = {
        'pcc_1': metrics['mel_real_pred']['pcc_1'],
        'pcc_2': metrics['mel_real_pred']['pcc_2'],
        'pcc_3': metrics['mel_real_pred']['pcc_3'],
        'pcc_4': metrics['mel_real_pred']['pcc_4'],
        'pcc_5': metrics['mel_real_pred']['pcc_5'],
        'pcc_6': metrics['mel_real_pred']['pcc_6'],
        'mse_1': metrics['mel_real_pred']['mse_1'],
        'mse_2': metrics['mel_real_pred']['mse_2'],
        'mse_3': metrics['mel_real_pred']['mse_3'],
        'mse_4': metrics['mel_real_pred']['mse_4'],
        'mse_5': metrics['mel_real_pred']['mse_5'],
        'mse_6': metrics['mel_real_pred']['mse_6']
        }

        metrics_all[f'fold_{fold}'] = metrics_save

        # 将每个指标的值添加到对应的列表中
        for key, value in metrics_save.items():
            metrics_mean[key].append(value)
            metrics_std[key].append(value)

    # 计算每个指标的平均值和标准差
    for key in metrics_mean:
        metrics_mean[key] = np.mean(metrics_mean[key])
        metrics_std[key] = np.std(metrics_std[key])
    metrics_all['mean'] = metrics_mean
    metrics_all['std'] = metrics_std

    #save metrics
    with open(os.path.join(orig_result_path, f'{dataset}/{patient}/metrics_multiscale.pkl'), 'wb') as f:
        pickle.dump(metrics_all,f)
        f.close()

    metrics_all_excel = pd.DataFrame.from_dict(metrics_all, orient='index')
    print(metrics_all_excel)
    metrics_all_excel.to_excel(os.path.join(orig_result_path, f'{dataset}/{patient}/metrics_multiscale.xlsx'), index=True)



def save_all_metrics_to_excel(dataset, orig_result_path):
    if dataset == 'sentence':
        pts = ['p%d'%i for i in range(1,4)]
    elif dataset == 'word':
        pts = ['sub-%02d'%i for i in range(1,11)]
    metrics_all = {}
    # metrics_mean = {'pcc_1': [], 'pcc_2': [], 'pcc_3': [], 'pcc_4': [], 'pcc_5': [], 'mse_1': [], 'mse_2': [], 'mse_3': [], 'mse_4': [], 'mse_5': []}
    # metrics_std = {'pcc_1': [], 'pcc_2': [], 'pcc_3': [], 'pcc_4': [], 'pcc_5': [], 'mse_1': [], 'mse_2': [], 'mse_3': [], 'mse_4': [], 'mse_5': []}
    
    metrics_mean = {'pcc_1': [], 'pcc_2': [], 'pcc_3': [], 'pcc_4': [], 'pcc_5': [], 'pcc_6': [], 'mse_1': [], 'mse_2': [], 'mse_3': [], 'mse_4': [], 'mse_5': [], 'mse_6': []}
    metrics_std = {'pcc_1': [], 'pcc_2': [], 'pcc_3': [], 'pcc_4': [], 'pcc_5': [], 'pcc_6': [], 'mse_1': [], 'mse_2': [], 'mse_3': [], 'mse_4': [], 'mse_5': [], 'mse_6': []}
    for pNr, pt in enumerate(pts):
        # result_path = f'results/{dataset}/{pt}'
        result_path = os.path.join(orig_result_path, f'{dataset}/{pt}')
        with open(os.path.join(result_path, 'metrics_multiscale.pkl'), 'rb') as f:
            metrics = pickle.load(f)
            f.close()

        # metrics_save = {
        # 'pcc_1': metrics['mean']['pcc_1'],
        # 'pcc_2': metrics['mean']['pcc_2'],
        # 'pcc_3': metrics['mean']['pcc_3'],
        # 'pcc_4': metrics['mean']['pcc_4'],
        # 'pcc_5': metrics['mean']['pcc_5'],
        # 'mse_1': metrics['mean']['mse_1'],
        # 'mse_2': metrics['mean']['mse_2'],
        # 'mse_3': metrics['mean']['mse_3'],
        # 'mse_4': metrics['mean']['mse_4'],
        # 'mse_5': metrics['mean']['mse_5']
        # }
        metrics_save = {
        'pcc_1': metrics['mean']['pcc_1'],
        'pcc_2': metrics['mean']['pcc_2'],
        'pcc_3': metrics['mean']['pcc_3'],
        'pcc_4': metrics['mean']['pcc_4'],
        'pcc_5': metrics['mean']['pcc_5'],
        'pcc_6': metrics['mean']['pcc_6'],
        'mse_1': metrics['mean']['mse_1'],
        'mse_2': metrics['mean']['mse_2'],
        'mse_3': metrics['mean']['mse_3'],
        'mse_4': metrics['mean']['mse_4'],
        'mse_5': metrics['mean']['mse_5'],
        'mse_6': metrics['mean']['mse_6']
        }

        metrics_all[pt] = metrics_save

        # 将每个指标的值添加到对应的列表中
        for key, value in metrics_save.items():
            metrics_mean[key].append(value)
            metrics_std[key].append(value)

    # 计算每个指标的平均值和标准差
    for key in metrics_mean:
        metrics_mean[key] = np.mean(metrics_mean[key])
        metrics_std[key] = np.std(metrics_std[key])
    metrics_all['mean'] = metrics_mean
    metrics_all['std'] = metrics_std

    metrics_all_excel = pd.DataFrame.from_dict(metrics_all, orient='index')
    print(metrics_all_excel)
    metrics_all_excel.to_excel(os.path.join(orig_result_path, f'{dataset}/metrics_all_multiscale.xlsx'), index=True)

if __name__ == '__main__':
    # work_dir = "/home/hyy/anaconda/stereoEEG2speech-master_ConvED-SR"
    work_dir = "/hdd/hyy/results"
    os.chdir(work_dir)

    # orig_result_path = 'results_multiscale_CBAM'
    orig_result_path = 'results_baseline'
    print(f'Calculate metrics for {orig_result_path}')

    dataset = 'word'
    pts = ['sub-%02d'%i for i in range(1,11)]
    for pNr, pt in   enumerate(pts):
        for fold in range(10): 
            calculate_metrics(dataset, pt, fold, orig_result_path)
        print(pt)
        save_singlesub_metrics_to_excel(dataset,pt,orig_result_path)
    save_all_metrics_to_excel(dataset,orig_result_path)

    # dataset = 'sentence'
    # pts = ['p%d'%i for i in range(1,4)]
    # for pNr, pt in enumerate(pts):
    #     # for fold in range(10):
    #     #     calculate_metrics(dataset, pt, fold, orig_result_path)
    #     save_singlesub_metrics_to_excel(dataset,pt,orig_result_path)
    # save_all_metrics_to_excel(dataset,orig_result_path)