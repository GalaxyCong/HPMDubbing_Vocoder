from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


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

    filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            x = np.load(os.path.join(a.input_mels_dir, filname))
            x = torch.FloatTensor(x).to(device)  # torch.Size([128, 80])
            x = x.permute(1,0).unsqueeze(0)  # torch.Size([1, 80, 128])
            y_g_hat = generator(x)  #  [1, 1, 20480])
            audio = y_g_hat.squeeze()  # torch.Size([20480])
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='/data/conggaoxiang/V2C/V2C_Code/example_V2C22050_256/MovieAnimation/mel')
    parser.add_argument('--output_dir', default='/data/conggaoxiang/V2C/V2C_Code/example_V2C16_HiFiGAN_E2_energyO_220/test_220')
    parser.add_argument('--checkpoint_file', default='/data/conggaoxiang/vocoder/hifi-gan-master/My16_test1_hifigan_V2C/g_00105000')
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

