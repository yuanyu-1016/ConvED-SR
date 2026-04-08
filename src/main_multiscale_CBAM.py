import warnings
warnings.filterwarnings("ignore")
from absl import logging, flags, app
import collections
import sh
import time
import random
import os
import torch 
import torch.nn.functional as F
import torchaudio.transforms as transforms
import torchaudio.functional as audioF
import pytorch_lightning as pl
import numpy as np
import sklearn
import IPython

import dataset as dataset
import models
from feat_extractor_CBAM import Feat_Extractor_Decoder
from vae import VAE

from torchcontrib.optim import SWA
from two_sample_distance import pdist

from speechbrain.inference.vocoders import HIFIGAN
import scipy.io.wavfile as wavfile

from calculate_metrics import test_metrics, calculate_metrics
from calflops import calculate_flops

flags.DEFINE_integer('batch_size', 256, '')
#flags.DEFINE_integer('hop', 10, 'eeg samples stride for train set')
flags.DEFINE_float('hop_in_ms', 25, 'eeg stride for train set [ms]')

# for debug
# flags.DEFINE_integer('batch_size', 2, '')
# flags.DEFINE_float('hop_in_ms', 100000, 'eeg stride for train set [ms]')

flags.DEFINE_string('optim', 'Adam', '')#AdamW
flags.DEFINE_float('learning_rate',5e-4, '')
# flags.DEFINE_float('learning_rate',1e-4, '')
# flags.DEFINE_float('learning_rate',5e-4, '')
flags.DEFINE_float('laplace_smoothing', 1e-2, 'for class weights, as a fraction of num_classes')#ignore
flags.DEFINE_float('teacher_forcing_ratio', 0.1, '')

flags.DEFINE_string('model_name', 'test', '')

flags.DEFINE_integer('gpus', 0, '')
# flags.DEFINE_integer('epochs', 50, '')
flags.DEFINE_integer('epochs', 1, '')
flags.DEFINE_integer('num_mel_centroids', 12, '')#ignore?
flags.DEFINE_integer('num_mel', 80, '')

flags.DEFINE_bool('debug', False, '')
flags.DEFINE_bool('clean_logs_dir', False, '')
flags.DEFINE_bool('final_test', True, '')
flags.DEFINE_bool('SWA', False, '') #True#False?
flags.DEFINE_integer('swa_start', 140, '')

flags.DEFINE_bool('MUILTISCALE', True, '')
flags.DEFINE_bool('DCAFormer', False, '')
flags.DEFINE_bool('PatchTST', False, '')
flags.DEFINE_bool('OLS', False, '')
flags.DEFINE_bool('DenseModel', False, '')#OLS False + DenseModel False = use RNNseq2seq

flags.DEFINE_integer('fold', 9, '')

flags.DEFINE_bool('use_wave_loss', False, '')
flags.DEFINE_bool('discretize_MFCCs', False, '')

flags.DEFINE_bool('mixed_loss', False, '')

flags.DEFINE_integer('seed', 224, '')

# flags.DEFINE_multi_integer('lr_milestones',[45],'epochs where lr is decreased.')
# flags.DEFINE_multi_integer('lr_milestones',[45,100,500,1000,1500,2000,3000],'epochs where lr is decreased.')
# flags.DEFINE_multi_integer('lr_milestones',[10,20,30,40,50,60,70],'epochs where lr is decreased.')
flags.DEFINE_multi_integer('lr_milestones',[20,40,60],'epochs where lr is decreased.')

FLAGS = flags.FLAGS

# 设置固定的随机种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main(_):
    work_dir = "/home/hyy/anaconda/stereoEEG2speech-master_ConvED-SR"
    results_dir = f"/hdd/hyy/results/results_{FLAGS.model_name}/{FLAGS.dataset}/{FLAGS.patient}/fold_{FLAGS.fold}"
    os.chdir(work_dir)
    os.makedirs(results_dir,exist_ok=True)

    if FLAGS.clean_logs_dir:
        sh.rm('-r', '-f', 'logs')
        sh.mkdir('logs')

    
    if not torch.cuda.is_available():
        FLAGS.gpus = 0
        torch.Tensor.cuda = lambda x: x

    if FLAGS.gpus:
        time.sleep(5)

    setup_seed(FLAGS.seed)

    # if not FLAGS.patient_eight:
    #     FLAGS.num_mel_centroids=10


    if FLAGS.OLS or FLAGS.DenseModel: #make sure output length is 1.
        assert FLAGS.use_MFCCs==True, "OLS can so far only be used with MFCCs"
        FLAGS.window_size=50
        print("Running with OLS/Dense. Re-setting window_size to 50ms")


    # k=5
    # for i in range(5):
    #     print(f'current fold{i}')
    #     train_ds = dataset.get_data(k, i, 'train')
    #     val_ds = dataset.get_data(k, i, 'val')
    #     test_ds = dataset.get_data(k, i, 'test')

    k=10
    train_ds = dataset.get_data(k=k, current_fold=FLAGS.fold, split='train', hop=FLAGS.hop_in_ms)
    val_ds = dataset.get_data(k=k, current_fold=FLAGS.fold, split='test')
    test_ds = val_ds
    # val_ds = dataset.get_data(k=k, current_fold=FLAGS.fold, split='val')
    # test_ds = dataset.get_data(k=k, current_fold=FLAGS.fold, split='test')

    # train_ds = dataset.get_data(split='train', hop=FLAGS.hop_in_ms)
    # val_ds = dataset.get_data(split='val')
    # test_ds = dataset.get_data(split='test')
    
    # for debug
    # val_ds = train_ds
    # test_ds = train_ds

    logging.info(f'train size: {len(train_ds)}, val size: {len(val_ds)}, test size: {len(test_ds)}')
    # num_classes = train_ds.num_audio_classes
    sampling_rate_audio = round(FLAGS.sampling_rate_eeg * val_ds.audio_eeg_sample_ratio)

    # if not FLAGS.use_MFCCs:
    #     class_freqs = torch.histc(train_ds.audio.float(), bins=num_classes, min=0, max=num_classes-1)
    #     class_freqs += FLAGS.laplace_smoothing * num_classes
    #     class_weights = 1. / class_freqs
    #     class_weights /= class_weights.sum()
    #     val_acc_for_mode = (train_ds.audio == val_ds.audio.mode()[0].item()).float().mean()
    #     logging.info(f'Validation accuracy when predicting mode: {val_acc_for_mode}')

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.input_shape = train_ds[0][0].shape # first batch of EEG data #seq_len_input, num_channels

            self.mel_transformer=val_ds.tacotron_mel_transformer

            self.val_outputs=[]
            self.test_outputs=[]
            
            if FLAGS.discretize_MFCCs:
                global_k_means_quantization=True # If True, centroids from kmeans across all bins are used. If False, centroids from within bins are used.
                if global_k_means_quantization:
                    self.mel_spec_discretizer=dataset.GlobalMelSpecDiscretizer()
                else:
                    self.mel_spec_discretizer=dataset.LocalMelSpecDiscretizer()

            if FLAGS.use_MFCCs:
                self.output_shape=self.mel_transformer.mel_spectrogram(train_ds[0][1].unsqueeze(0)).squeeze(0).T.shape #seq_len x mel bins ??
                if FLAGS.discretize_MFCCs:
                    class_weights=None
                    #class_weights=torch.load('class_weights_median.pt')
                    if class_weights is None:
                        self.criterion=torch.nn.CrossEntropyLoss(reduction='none') #combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
                    else:
                        self.criterion=torch.nn.CrossEntropyLoss(weight=class_weights,reduction='none') 
                else:
                    self.criterion = torch.nn.MSELoss(reduction='none')

            # else:
            #     self.output_shape = train_ds[0][1].shape + (num_classes,) #first batch of audio data
            #     self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
            if FLAGS.MUILTISCALE:
    
                self.model = Feat_Extractor_Decoder(channel_in=self.input_shape[1], hidden=256, latent_channels=256)
            
            #load hifi_gan.
            self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")
            self.hifi_gan.device = f'cuda:{FLAGS.gpus}'
        def shuffle_feats(self, input_tensor):
            """
            对输入张量的 feats 维度进行随机打乱。
            
            Args:
                input_tensor (torch.Tensor): 输入张量,形状为 (batch_size, time, feats)。
            
            Returns:
                torch.Tensor: 打乱 feats 维度后的张量。
            """
            # 获取 feats 维度的索引
            feats_dim = 2
            
            # 创建一个随机排列的索引
            perm_idx = torch.randperm(input_tensor.size(feats_dim)).to(input_tensor.device)
            
            # 使用重排后的索引来重排 feats 维度
            shuffled_input = input_tensor.index_select(feats_dim, perm_idx)
            
            return shuffled_input
        
        def scale_feats(self, input_tensor, a_min, a_max):
            """
            对输入张量的 time 维度和 feats 维度随机乘比例。
            
            Args:
                input_tensor (torch.Tensor): 输入张量,形状为 (batch_size, time, feats)。
                a_min (float): 随机乘比例的最小值。
                a_max (float): 随机乘比例的最大值。
            
            Returns:
                torch.Tensor: 缩放后的张量。
            """
            
            # 为每个 feats 维度创建一个随机乘比例张量
            feat_scale = torch.rand(1, 1, input_tensor.size(2), device=input_tensor.device) * (a_max - a_min) + a_min
            
            # 应用缩放操作
            scaled_input = input_tensor * feat_scale
            
            return scaled_input

        def loss(self, logits, y):
            if not FLAGS.use_MFCCs:
                y = y.flatten() # [bs x seq_len] -> [bs*seq_len] @YK: WHY??
                logits = logits.flatten(0, 1) #[bs x seq_len x num_classes] -> [bs*seq_len x num_classes]
            if FLAGS.discretize_MFCCs:
                current_batch_size=logits.shape[0] #will be flags.batchsize almost always but not in the very end of validation/train
                logits=logits.reshape((current_batch_size,-1,FLAGS.num_mel,FLAGS.num_mel_centroids)) # bs x seq_len x 80 x 12
                logits=logits.transpose(1,3).transpose(2,3)
            return self.criterion(logits, y) #[bs * seq_len]
        def l2_regularization_loss(self, model, lambda_reg=1):
            """
            计算 L2 正则化损失
            
            参数:
            model (nn.Module) - 需要计算正则化损失的模型
            lambda_reg (float) - 正则化系数,控制正则化的强度
            
            返回:
            reg_loss (torch.Tensor) - L2 正则化损失
            """
            reg_loss = 0
            for param in model.parameters():
                reg_loss += torch.sum(param ** 2)
            reg_loss *= lambda_reg / 2
            return reg_loss
        def kl_loss(self, mu, logvar):
            return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

        def feats_loss(self, feats1, feats2):
            loss = 0
            for i in range(len(feats1)):
                loss += F.mse_loss(feats1[i], feats2[i])
            return loss
        def WeightedCosineSimilarityLoss(self, input, target):
            '''
            计算加权余弦相似度损失
            
            参数:
            input (torch.Tensor): 模型输出的预测向量, 形状为 (batch_size, feature_dim)
            target (torch.Tensor): 目标向量, 形状为 (batch_size, feature_dim)
            
            返回:
            loss (torch.Tensor): 加权余弦相似度损失
            '''
            # 计算点积
            dot_product = torch.sum(input * target, dim=1)
            
            # 计算向量范数
            input_norm = torch.norm(input, dim=1)
            target_norm = torch.norm(target, dim=1)
            
            # 计算加权余弦相似度
            weighted_cos_sim = dot_product / (input_norm * target_norm)
            
            # 计算损失
            loss = 1 - weighted_cos_sim
            
            return loss.mean()

        def logits_to_classes(self,logits):
            current_batch_size=logits.shape[0] #will be flags.batchsize almost always but not in the very end of validation/train
            logits=logits.reshape((current_batch_size,-1,FLAGS.num_mel,FLAGS.num_mel_centroids)) # bs x seq_len x 80 x 12
            logits=torch.argmax(logits,dim=3) #bs x seq_len x 80. each entry is the class.
            return logits

        def logits_to_mel_centroids(self,logits):
            logits=self.logits_to_classes(logits)
            return self.mel_spec_discretizer.class_to_centroids(logits)

        def contrastive_loss(self, x, encoder_outputs):
            #encoder_outputs [bs x seq_len_after_conv x hidden_size * directions] 
            #x [bs x seq_len_before_conv x channels]

            patient_4=(encoder_outputs[x[:,0,214]==0,:,:])
            patient_8=(encoder_outputs[x[:,0,214]==1,:,:])

            if len(patient_4<FLAGS.batch_size) and len(patient_8<FLAGS.batch_size):
                distance_within_patient_4=torch.mean(torch.nn.functional.pdist(patient_4.flatten(1,2))) #the flatten is not nice but very fast
                distance_within_patient_8=torch.mean(torch.nn.functional.pdist(patient_8.flatten(1,2)))
                distance_across_patients=torch.mean(pdist(patient_4.flatten(1,2),patient_8.flatten(1,2)))  #(n_1, d),(n_2,d)
                delta=1
                return (distance_within_patient_4+distance_within_patient_8+distance_across_patients)**2
            else:
                return 0

        def accuracy(self, logits, y, topk=1):
            if FLAGS.use_MFCCs: # y and logits are [bs x num_frames x num_bins]
                if FLAGS.discretize_MFCCs:
                    #1. actual accrutacy
                    class_predictions=self.logits_to_classes(logits)
                    accuracy=1.0*torch.sum(y==class_predictions)/y.numel()

                    #2. soft (differentiable) pearson r:
                    if FLAGS.mixed_loss:
                        current_batch_size=logits.shape[0] #will be flags.batchsize almost always but not in the very end of validation/train
                        logits_in_softmax=logits.reshape((current_batch_size,-1,FLAGS.num_mel,FLAGS.num_mel_centroids)) # bs x seq_len x 80 x 12
                        logits_in_softmax=torch.nn.functional.softmax(logits_in_softmax,dim=-1) # bs x seq_len x 80 x 12
                        y_in_softmax=torch.nn.functional.one_hot(y).float()

                        logits_in_softmax=logits_in_softmax.flatten(2,3)
                        y_in_softmax=y_in_softmax.flatten(2,3)
                        soft_pearson_r=torch.nn.functional.cosine_similarity(y_in_softmax-torch.mean(y_in_softmax,dim=1).unsqueeze(1),logits_in_softmax-torch.mean(logits_in_softmax,dim=1).unsqueeze(1),dim=1)   
                    else:
                        soft_pearson_r=accuracy.new_zeros((1))

                    #3. pearson r:
                    logits=self.logits_to_mel_centroids(logits)
                    y=self.mel_spec_discretizer.class_to_centroids(y)
                    pearson_r=torch.nn.functional.cosine_similarity(y-torch.mean(y,dim=1).unsqueeze(1),logits-torch.mean(logits,dim=1).unsqueeze(1),dim=1)   


                    return pearson_r,accuracy, soft_pearson_r #mean is taken later
                else:
                    pearson_r=torch.nn.functional.cosine_similarity(y-torch.mean(y,dim=1).unsqueeze(1),logits-torch.mean(logits,dim=1).unsqueeze(1),dim=1)   
                    return pearson_r
            else:
                _, topi = torch.topk(logits, k=topk, dim=-1)
                return (y.unsqueeze(-1) == topi).float().sum(-1)

        def forward(self, x):
            pass

        def training_step(self, batch, batch_idx):
            if self.trainer.current_epoch>=FLAGS.epochs-2 and FLAGS.SWA:
                for param_group in self.trainer.optimizers[0].param_groups:
                            param_group['lr'] = 0.000000000000000001 #precent sgd from walking away from averaged point

            x, y_wave = batch


            pitch = audioF.detect_pitch_frequency(y_wave, 22050)

            if FLAGS.use_MFCCs:
                y=self.mel_transformer.mel_spectrogram(y_wave).transpose(1,2)
                if FLAGS.discretize_MFCCs:
                    y=self.mel_spec_discretizer.mel_to_class(y_wave)
            teacher_forcing = torch.bernoulli(x.new_ones((x.shape[0],)) * FLAGS.teacher_forcing_ratio).byte()
           

            if FLAGS.MUILTISCALE:
                logits = self.model(x)

                logits0 = logits[0]
                logits1 = logits[1]
                logits2 = logits[2]
                logits = logits[3]
                
                y2 = F.interpolate(y.unsqueeze(1), scale_factor=0.5).squeeze(1)
                y1 = F.interpolate(y2.unsqueeze(1), scale_factor=0.5).squeeze(1)
                y0 = F.interpolate(y1.unsqueeze(1), scale_factor=0.5).squeeze(1)
                mel_loss0 = self.loss(logits0, y0).mean()
                mel_loss1 = self.loss(logits1, y1).mean()
                mel_loss2 = self.loss(logits2, y2).mean()
                mel_loss3 = self.loss(logits, y).mean()
                mel_loss = mel_loss3

                loss = mel_loss0 + mel_loss1 + mel_loss2 + mel_loss3

            elif FLAGS.DCAFormer or FLAGS.PatchTST:
                logits = self.model(x)
            else:
                logits,attn_matrix, encoder_outputs = self.model(x, y=y, teacher_forcing=teacher_forcing)
                
                mel_loss = self.loss(logits, y).mean()
                loss = mel_loss



            if FLAGS.use_wave_loss:
                rec_wave = self.hifi_gan.decode_batch(logits.transpose(1,2)).squeeze(1)
                if rec_wave.shape[-1]!=y_wave.shape[-1]:
                    cutLen = int((rec_wave.shape[-1]-y_wave.shape[-1])/2)
                    rec_wave = rec_wave[:,cutLen:cutLen+y_wave.shape[-1]]
                loss_wave = self.loss(rec_wave, y_wave)


            # if FLAGS.double_trouble:
            #     contrastive_loss=self.contrastive_loss(x,encoder_outputs)
            #     loss= loss + 0.1*contrastive_loss

            if FLAGS.discretize_MFCCs:
                pearson_r,acc, soft_pearson_r= self.accuracy(logits, y) #first is pearson_r
            else:
                acc = self.accuracy(logits, y)

            if not FLAGS.use_MFCCs:
                acc5 = self.accuracy(logits, y, 5)

            if FLAGS.MUILTISCALE: 
                for param_group in self.trainer.optimizers[0].param_groups:
                    current_lr=(param_group['lr'])
                logs = {'pearson_r/train': acc.mean(), 'loss/train': loss.mean(), 'mel_loss/train': mel_loss.mean(), 
                        'learning_rate': current_lr}
                self.logger.experiment.add_scalar('pearson_r/train', acc.mean(), self.global_step)
                self.logger.experiment.add_scalar('loss/train', loss.mean(), self.global_step)
                self.logger.experiment.add_scalar('mel_loss/train', mel_loss.mean(), self.global_step)
                self.logger.experiment.add_scalar('learning_rate', current_lr, self.global_step)
                if batch_idx == 0:
                    if FLAGS.discretize_MFCCs:
                        logits=self.logits_to_mel_centroids(logits)
                        y=self.mel_spec_discretizer.class_to_centroids(y)
                    MFCC_plot = dataset.create_MFCC_plot(logits[0],y[0]) #passt first sample
                    self.logger.experiment.add_image('MFCC_plot/train', MFCC_plot, global_step=self.global_step, dataformats='HWC')
            elif FLAGS.use_MFCCs: 
                for param_group in self.trainer.optimizers[0].param_groups:
                    current_lr=(param_group['lr'])
                if FLAGS.discretize_MFCCs:
                    logs = {'loss/train': loss.mean(), 'pearson_r/train': pearson_r.mean(), 'accuracy/train': acc.mean(), 'learning_rate': current_lr}
                elif FLAGS.use_wave_loss:
                    logs = {'loss/train': loss.mean(),'loss_wave/train': loss_wave.mean(), 'pearson_r/train': acc.mean(), 'learning_rate': current_lr}
                    self.logger.experiment.add_scalar('loss/train', loss.mean(), self.global_step)
                    self.logger.experiment.add_scalar('loss_wave/train', loss_wave.mean(), self.global_step)
                    self.logger.experiment.add_scalar('pearson_r/train', acc.mean(), self.global_step)
                    self.logger.experiment.add_scalar('learning_rate', current_lr, self.global_step)
                else:
                    logs = {'loss/train': loss.mean(), 'pearson_r/train': acc.mean(), 'learning_rate': current_lr}
                    self.logger.experiment.add_scalar('loss/train', loss.mean(), self.global_step)
                    self.logger.experiment.add_scalar('mel_loss/train', mel_loss.mean(), self.global_step)
                    self.logger.experiment.add_scalar('pearson_r/train', acc.mean(), self.global_step)
                    self.logger.experiment.add_scalar('learning_rate', current_lr, self.global_step)
                if batch_idx == 0:
                    if FLAGS.discretize_MFCCs:
                        logits=self.logits_to_mel_centroids(logits)
                        y=self.mel_spec_discretizer.class_to_centroids(y)
                    MFCC_plot = dataset.create_MFCC_plot(logits[0],y[0]) #passt first sample
                    self.logger.experiment.add_image('MFCC_plot/train', MFCC_plot, global_step=self.global_step, dataformats='HWC')
                    if not (FLAGS.MUILTISCALE or FLAGS.DCAFormer or FLAGS.PatchTST):
                        attn_plot = models.create_attention_plot(attn_matrix)
                        self.logger.experiment.add_image('Attention_Matrix', attn_plot, global_step=self.global_step, dataformats='HWC')

            else:
                logs = {'loss/train': loss.mean(), 'acc/acc': acc.mean(), 'acc5': acc5.mean()}
                if batch_idx == 0:
                    audio_idx = random.randint(0, len(batch) - 1)
                    audio_real = dataset.audio_classes_to_signal_th(y[audio_idx])
                    audio_pred = dataset.audio_classes_to_signal_th(logits[audio_idx].argmax(-1))
                    self.logger.experiment.add_audio(
                            tag='audio_real',
                            snd_tensor=audio_real.unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=sampling_rate_audio,
                            )
                    self.logger.experiment.add_audio(
                            tag='audio_pred',
                            snd_tensor=audio_pred.unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=sampling_rate_audio,
                            )

                    audio_plot = dataset.create_audio_plot([
                            (audio_real, 'real'),
                            (audio_pred, 'predicted_tf_{}'.format(teacher_forcing[audio_idx].item())),
                            ])
                    self.logger.experiment.add_image('audio_plot', audio_plot, global_step=self.global_step, dataformats='HWC')
            #return {**logs, 'log': logs}
            if FLAGS.mixed_loss:
                my_lambda=0.75
                return {'loss': my_lambda* logs['loss/train'] - (1-my_lambda)*soft_pearson_r.mean(), 'log': logs}
            elif FLAGS.use_wave_loss:
                my_lambda=0.1
                return {'loss': logs['loss/train']+my_lambda*logs['loss_wave/train'] , 'log': logs}
            else:
                return {'loss': logs['loss/train'], 'log': logs}

        def on_train_end(self):
            # print('Training is done. Saving model...')
            torch.save(self.model.state_dict(), os.path.join(results_dir, 'model.pt'))

        def validation_step(self, batch, batch_idx):
            x, y_wave = batch
            if FLAGS.use_MFCCs:
                y=self.mel_transformer.mel_spectrogram(y_wave).transpose(1,2)
                if FLAGS.discretize_MFCCs:
                    y=self.mel_spec_discretizer.mel_to_class(y) # bs x seq_len (num mel frames) x mel_bins
            if FLAGS.MUILTISCALE:
                logits = self.model(x)
                logits = logits[3]
            elif FLAGS.DCAFormer or FLAGS.PatchTST:
                logits = self.model(x)
            else:
                logits,attn_matrix,encoder_outputs = self.model(x)  # no teacher forcing

            mel_loss = self.loss(logits, y)

            loss = mel_loss

            # loss = self.loss(logits, y)
            if FLAGS.discretize_MFCCs:
                pearson_r,acc,soft_pearson_r = self.accuracy(logits, y) # in OLS/DenseNet the MFCC frames are not a sequence. 
            else:
                acc= self.accuracy(logits,y)
            acc5=acc.new_zeros((1,1))

            rec_wave = self.hifi_gan.decode_batch(logits.transpose(1,2)).squeeze(1)
            yhat_wave = self.hifi_gan.decode_batch(y.transpose(1,2)).squeeze(1)
            if rec_wave.shape[-1]!=y_wave.shape[-1]:
                cutLen = int((rec_wave.shape[-1]-y_wave.shape[-1])/2)
                rec_wave = rec_wave[:,cutLen:cutLen+y_wave.shape[-1]]
                yhat_wave = yhat_wave[:,cutLen:cutLen+y_wave.shape[-1]]

            # loss_wave_yhat_rec = self.loss(rec_wave, yhat_wave)
            loss_wave_y_rec = self.loss(rec_wave, y_wave)

            if FLAGS.MUILTISCALE:  
                every_kth=1#int(1024/256)
                outs = {'loss': loss, 'mel_loss': mel_loss, 'acc': acc, 'acc5': acc5,
                         'targets': y[:,::every_kth,:],'predictions': logits[:,::every_kth,:]}
                self.logger.experiment.add_scalar('loss/val', loss.mean(), self.global_step)
                self.logger.experiment.add_scalar('mel_loss/val', mel_loss.mean(), self.global_step)
                self.logger.experiment.add_scalar('loss_wave_y_rec/val', loss_wave_y_rec.mean(), self.global_step)
            elif FLAGS.use_MFCCs: 
                every_kth=1#int(1024/256)
                if FLAGS.discretize_MFCCs:
                    y=self.mel_spec_discretizer.class_to_centroids(y)
                    logits=self.logits_to_mel_centroids(logits)
                    outs = {'loss': loss, 'acc': pearson_r, 'acc5': acc5, 'targets': y[:,::every_kth,:],'predictions': logits[:,::every_kth,:],'actual_acc': acc.unsqueeze(0)} #unsq. for consistency in later cat.
                else:
                    outs = {'loss': loss, 'mel_loss': mel_loss, 'acc': acc, 'acc5': acc5, 'targets': y[:,::every_kth,:],'predictions': logits[:,::every_kth,:]}
                    # self.logger.experiment.add_scalar('loss_wave_yhat_rec/val', loss_wave_yhat_rec.mean(), self.global_step)
                    self.logger.experiment.add_scalar('loss_wave_y_rec/val', loss_wave_y_rec.mean(), self.global_step)
                    if self.global_step == 0:
                        for i in range(1):
                            self.logger.experiment.add_audio('generated/y_{}'.format(i), y_wave[i], global_step=self.global_step, sample_rate=22050)
                            self.logger.experiment.add_audio('generated/y_hat_{}'.format(i), yhat_wave[i], global_step=self.global_step, sample_rate=22050)
                    for i in range(1):
                        self.logger.experiment.add_audio('generated/y_rec_{}'.format(i), rec_wave[i], global_step=self.global_step, sample_rate=22050)
                #if batch_idx == 0:
                 #   outs['MFCC_plot_val'] = dataset.create_MFCC_plot(logits[0],y[0])
            else:
                outs = {'loss': loss, 'acc': acc, 'acc5': acc5}
                if batch_idx == 0:
                    audio_idx = random.randint(0, len(batch) - 1)
                    outs['val_audio_real'] = dataset.audio_classes_to_signal_th(y[audio_idx])
                    outs['val_audio_pred'] = dataset.audio_classes_to_signal_th(logits[audio_idx].argmax(-1))
                    outs['val_audio_plot'] = dataset.create_audio_plot([
                            (outs['val_audio_real'], 'real'),
                            (outs['val_audio_pred'], 'predicted'),
                            ])
            return outs
        
        def on_validation_batch_end(self, out, batch, batch_idx):
            # 将 out 添加到 self.outputs 中
            self.val_outputs.append(out)

        def on_validation_epoch_end(self): #outputs contains outs of validation_step for entire val set.
            outs = collections.defaultdict(list)
            for o in self.val_outputs:
                for k, v in o.items(): 
                    outs[k].append(v)

            outputs = {k: torch.cat(v, 0) for k, v in outs.items()}
            self.val_outputs = []
            #acc = outputs['acc'].mean()

            all_predictions=outputs['predictions'].view((outputs['predictions'].shape[0]*outputs['predictions'].shape[1],outputs['predictions'].shape[2]))
            all_targets=outputs['targets'].view((outputs['targets'].shape[0]*outputs['targets'].shape[1],outputs['targets'].shape[2]))            
            if FLAGS.discretize_MFCCs: #acc function cannot handle this input if discretization is used.
                acc=torch.nn.functional.cosine_similarity(all_predictions-torch.mean(all_predictions,dim=1).unsqueeze(1),all_targets-torch.mean(all_targets,dim=1).unsqueeze(1),dim=1)   
                acc=acc.mean()
            else:
                acc_all= self.accuracy(all_predictions.unsqueeze(0),all_targets.unsqueeze(0)).mean()

            acc5 = outputs['acc5'].mean()
            if FLAGS.discretize_MFCCs:
                actual_acc = outputs['actual_acc'].mean()

                #torch.save(all_predictions.T,f"mel_preds/EPOCH50bs_{FLAGS.batch_size}_lr_{FLAGS.learning_rate}_tfr_{FLAGS.teacher_forcing_ratio}_ws_{FLAGS.window_size}_emb_dm_{FLAGS.hidden_size}.wav.pt")
            # if self.trainer.current_epoch==FLAGS.epochs-2:
            #     os.makedirs("mel_preds", exist_ok=True)
            #     torch.save(all_predictions.T,f"mel_preds/bs_{FLAGS.batch_size}_lr_{FLAGS.learning_rate}_tfr_{FLAGS.teacher_forcing_ratio}_ws_{FLAGS.window_size}_emb_dm_{FLAGS.hidden_size}_do_{FLAGS.dropout}_pnpnd_{FLAGS.pre_and_postnet_dim}.wav.pt")
            #     torch.save(all_targets.T,f"mel_preds/TARGETS_bs_{FLAGS.batch_size}_lr_{FLAGS.learning_rate}_tfr_{FLAGS.teacher_forcing_ratio}_ws_{FLAGS.window_size}_emb_dm_{FLAGS.hidden_size}_do_{FLAGS.dropout}_pnpnd_{FLAGS.pre_and_postnet_dim}.wav.pt")

            #     if FLAGS.SWA:
            #         self.trainer.optimizers[0].swap_swa_sgd()
            #         print("SWITCHING TO SWA WEIGHTS")
            #IPython.embed()
            #torch.save(all_targets.T,'mel_preds/new_targets'+ str(self.trainer.current_epoch))
            if self.trainer.current_epoch==FLAGS.epochs-1 and FLAGS.SWA:
                torch.save(all_predictions.T,f"mel_preds/AFTER_SWA_bs_{FLAGS.batch_size}_lr_{FLAGS.learning_rate}_tfr_{FLAGS.teacher_forcing_ratio}_ws_{FLAGS.window_size}_emb_dm_{FLAGS.hidden_size}.wav.pt")


            if FLAGS.use_MFCCs:
                self.logger.experiment.add_image('MFCC_plot/val', dataset.create_MFCC_plot(all_predictions[0:400,:],all_targets[0:400,:]), global_step=self.global_step, dataformats='HWC')
                self.logger.experiment.add_image('MFCC_plot/val_all', dataset.create_MFCC_plot(all_predictions,all_targets), global_step=self.global_step, dataformats='HWC')
                # self.logger.experiment.add_image('MFCC_plot/val', dataset.create_MFCC_plot(all_predictions,all_targets), global_step=self.global_step, dataformats='HWC')
            else:
                for tag in ('val_audio_real', 'val_audio_pred'):
                    self.logger.experiment.add_audio(
                            tag=tag,
                            snd_tensor=outputs[tag].unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=sampling_rate_audio,
                            )
                self.logger.experiment.add_image('val_audio_plot', outputs['val_audio_plot'], global_step=self.global_step, dataformats='HWC')

            if FLAGS.use_MFCCs: 
                if FLAGS.discretize_MFCCs:
                    logs = {
                            'loss/val': outputs['loss'].mean(),
                            'pearson_r/val': acc,
                            'accuracy/val':actual_acc
                            }
                else:
                    logs = {
                        'loss/val': outputs['loss'].mean(),
                        'pearson_r/val': outputs['acc'].mean(),
                        'pearson_r_all/val': acc_all,
                        }
                    self.logger.experiment.add_scalar('loss/val', outputs['loss'].mean(), self.global_step)
                    self.logger.experiment.add_scalar('mel_loss/val', outputs['mel_loss'].mean(), self.global_step)
                    self.logger.experiment.add_scalar('pearson_rl/val', outputs['acc'].mean(), self.global_step)
                    self.logger.experiment.add_scalar('pearson_r_all/val', acc_all, self.global_step)
            # else:
            #     logs = {
            #             'loss/val': outputs['loss'].mean(),
            #             'acc/val_acc': acc,
            #             'val_acc_rel_vs_rand': acc / (1. / num_classes),
            #             'val_acc_rel_vs_mode': acc / val_acc_for_mode,
            #             'val_acc5': acc5,
            #             'val_acc5_rel_vs_rand': acc5 / (1. / num_classes),
            #             'val_acc5_rel_vs_mode': acc5 / val_acc_for_mode,
            #             }
            return {'val_loss': logs['loss/val'], 'log': logs}

        def on_test_start(self):
            # print('Load model...')
            self.model.load_state_dict(torch.load(os.path.join(results_dir, 'model.pt')))

        def test_step(self, batch, batch_idx, dataloader_idx=0):

            x, y_wave = batch
            if FLAGS.use_MFCCs:
                y=self.mel_transformer.mel_spectrogram(y_wave).transpose(1,2)
                if FLAGS.discretize_MFCCs:
                    y=self.mel_spec_discretizer.mel_to_class(y_wave)

            if FLAGS.MUILTISCALE:
                logits = self.model(x)
                logits = logits[3]
            elif FLAGS.DCAFormer or FLAGS.PatchTST:
                logits = self.model(x)
            else:
                logits,attn_matrix,encoder_outputs = self.model(x)  # no teacher forcing
            # preds = logits.argmax(-1)
            # audio_real = dataset.audio_classes_to_signal_th(y).flatten()
            # audio_pred = dataset.audio_classes_to_signal_th(preds).flatten()
            yhat_wave = self.hifi_gan.decode_batch(y.transpose(1,2)).squeeze(1)
            rec_wave = self.hifi_gan.decode_batch(logits.transpose(1,2)).squeeze(1)
            if rec_wave.shape[-1]!=y_wave.shape[-1]:
                cutLen = int((rec_wave.shape[-1]-y_wave.shape[-1])/2)
                rec_wave = rec_wave[:,cutLen:cutLen+y_wave.shape[-1]]
                yhat_wave = yhat_wave[:,cutLen:cutLen+y_wave.shape[-1]]
            
            return {'mel_spec_real': y,
                    'mel_spec_pred': logits,
                    'audio_real': y_wave, 
                    'audio_rec': yhat_wave,
                    'audio_pred': rec_wave}
        
        def on_test_batch_end(self, out, batch, batch_idx, dataloader_idx=0):
            # 将 out 添加到 self.outputs 中
            self.test_outputs.append(out)

        def on_test_epoch_end(self):
            # for outputs in self.test_outputs:
            #     audio_real = torch.cat([o['audio_real'] for o in outputs], 0).cpu().numpy()
            #     audio_pred = torch.cat([o['audio_pred'] for o in outputs], 0).cpu().numpy()
            #     # TODO: save audio

            #     os.makedirs("{FLAGS.result_root}",exist_ok=True)
            #     np.save(f'{FLAGS.result_root}/test_bs_{FLAGS.batch_size}',np.concatenate((audio_real,audio_pred)))
            outs = collections.defaultdict(list)
            for o in self.test_outputs:
                for k, v in o.items(): 
                    outs[k].append(v)

            outputs = {k: torch.cat(v, 0) for k, v in outs.items()}
            mel_spec_real = outputs['mel_spec_real'].view((outputs['mel_spec_real'].shape[0]*outputs['mel_spec_real'].shape[1],outputs['mel_spec_real'].shape[2])).cpu().numpy()
            mel_spec_pred = outputs['mel_spec_pred'].view((outputs['mel_spec_pred'].shape[0]*outputs['mel_spec_pred'].shape[1],outputs['mel_spec_pred'].shape[2])).cpu().numpy()
            audio_real = outputs['audio_real'].view((outputs['audio_real'].shape[0]*outputs['audio_real'].shape[1])).cpu().numpy()
            audio_rec = outputs['audio_rec'].view((outputs['audio_rec'].shape[0]*outputs['audio_rec'].shape[1])).cpu().numpy()
            audio_pred = outputs['audio_pred'].view((outputs['audio_pred'].shape[0]*outputs['audio_pred'].shape[1])).cpu().numpy()
            self.test_outputs = []

            np.save(os.path.join(results_dir, 'mel_spec_real'), mel_spec_real)
            np.save(os.path.join(results_dir, 'mel_spec_pred'), mel_spec_pred)
            wavfile.write(os.path.join(results_dir, 'audio_real.wav'), 22050, audio_real)
            wavfile.write(os.path.join(results_dir, 'audio_rec.wav'), 22050, audio_rec)
            wavfile.write(os.path.join(results_dir, 'audio_pred.wav'), 22050, audio_pred)

            calculate_metrics(FLAGS.dataset, FLAGS.patient, FLAGS.fold, f"/hdd/hyy/results/results_{FLAGS.model_name}")

            return {} # has to be here

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                    train_ds, 
                    shuffle=True,
                    drop_last=True,
                    num_workers=3,
                    batch_size=FLAGS.batch_size)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(
                    val_ds, 
                    shuffle=False,
                    drop_last=False,
                    num_workers=3,
                    batch_size=FLAGS.batch_size)

        def test_dataloader(self):
            return torch.utils.data.DataLoader(
                    test_ds, 
                    shuffle=False,
                    drop_last=False,
                    num_workers=3,
                    batch_size=FLAGS.batch_size)
            # train_ds_full_hop = dataset.get_data('train')
            # return [torch.utils.data.DataLoader(ds, shuffle=False, drop_last=False, num_workers=3,
            #         batch_size=FLAGS.batch_size) for ds in (train_ds_full_hop, val_ds)]

        def configure_optimizers(self): 
            optim = next(o for o in dir(torch.optim) if o.lower() == FLAGS.optim.lower()) # "Adam"
            optimizer=getattr(torch.optim, optim)(self.parameters(), lr=FLAGS.learning_rate) # optimizer object
            #optimizer=torch.optim.SGD(self.parameters(), lr=FLAGS.learning_rate,weight_decay=0.00001) 
            optimizer=torch.optim.AdamW(self.parameters(),lr=FLAGS.learning_rate,weight_decay=0.01)
            if FLAGS.SWA:
                iterations_per_epoch=int(len(train_ds)/FLAGS.batch_size)
                optimizer = SWA(optimizer, swa_start=int(FLAGS.swa_start*iterations_per_epoch), swa_freq=50, swa_lr=FLAGS.learning_rate/10)
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=FLAGS.lr_milestones, gamma=0.5)
            #scheduler=torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=FLAGS.learning_rate/2,max_lr=2*FLAGS.learning_rate,step_size_up=2,step_size_down=2,cycle_momentum=False)
            #step_size_up is in epochs! I don't know why the hell
            return {'optimizer':optimizer,'lr_scheduler':scheduler}

    model = Model()

    #run_on_gpu(cpu_allowed=True)

    trainer = pl.Trainer(
            devices=[FLAGS.gpus],
            accelerator='gpu',
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            default_root_dir='/hdd/hyy/results/logs',
            logger=pl.loggers.TensorBoardLogger('/hdd/hyy/results/logs',name=f"{FLAGS.model_name}/{FLAGS.dataset}/{FLAGS.patient}/fold_{FLAGS.fold}/bs {FLAGS.batch_size}, lr {FLAGS.learning_rate}, tfr {FLAGS.teacher_forcing_ratio}, ws {FLAGS.window_size}, lyrs {FLAGS.n_layers}, emb dm {FLAGS.hidden_size}, drpt {FLAGS.dropout},pnpnd {FLAGS.pre_and_postnet_dim} "),
            # terminate_on_nan=True,
            log_every_n_steps=100,
            num_sanity_val_steps=8,
            gradient_clip_val=2
            )

    trainer.fit(model)
    if FLAGS.final_test:
        trainer.test(model)

    # model =Feat_Extractor_Decoder(channel_in=110, hidden=256, latent_channels=256).to('cuda:0')
    # batch_size = 1
    # input_shape = (batch_size, 1024, 110)
    # # flops, macs, params = calculate_flops(model=model, 
    # #                                     input_shape=input_shape,
    # #                                     output_as_string=True,
    # #                                     output_precision=4)
    # # print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    
    # input_tensor = torch.randn(input_shape).to('cuda:0')

    # # 预热
    # model.eval()
    # for _ in range(10):
    #     with torch.no_grad():
    #         model(input_tensor)

    # # 测量前向传播时间
    # num_iterations = 100
    # start_time = time.time()
    # for _ in range(num_iterations):
    #     with torch.no_grad():
    #         logits = model(input_tensor)
    # end_time = time.time()

    # # 计算平均时间
    # # 计算平均时间并转换为毫秒
    # average_time_seconds = (end_time - start_time) / num_iterations
    # average_time_milliseconds = average_time_seconds * 1000
    # print(f"前向传播平均时间: {average_time_milliseconds:.6f}毫秒")


if __name__ == '__main__':
    app.run(main)
