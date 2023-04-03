from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import glob

import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
# 1024, 80, 16000, 160, 640, 0, 8000
# mel_spectrogram(x, 1024, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    """
    LJ-Speech_Demo
    """
    input_validation_file = '/data/conggaoxiang/vocoder/hifi-gan-master/LJSpeech-1.1/validation.txt'
    input_wavs_dir = "/data/conggaoxiang/vocoder/hifi-gan-master/LJSpeech-1.1/wavs16"
    with open(input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    filelist = validation_files

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            # wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav, sr = load_wav(filname)
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            # output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            output_file = os.path.join(a.output_dir, filname.split('/')[-1])
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_wavs_dir', default='/data/conggaoxiang/jyt/noise_16000_3320/chem')
    # parser.add_argument('--output_dir', default='/data/conggaoxiang/jyt/noise_16000_3320_hifigan')
    parser.add_argument('--input_wavs_dir', default='/data/conggaoxiang/vocoder')
    parser.add_argument('--output_dir', default='/data/conggaoxiang/vocoder/hifi-gan-master/checkpoint_hifigan_offical/My_MOD_MPD_16KHz')  # MOD_V1
    parser.add_argument('--checkpoint_file', default='/data/conggaoxiang/vocoder/hifi-gan-master/My_MOD_MPD_16KHz_Repeat2/g_02060000')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

