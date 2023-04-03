import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import audio as Audio
from text import _clean_text
import numpy as np
import librosa
from pathlib import Path
from scipy.io.wavfile import write
from joblib import Parallel, delayed


import tgt
import pyworld as pw
from preprocessors.utils import remove_outlier, get_alignment, average_by_duration
from scipy.interpolate import interp1d
import json
# output_folder '/data/conggaoxiang/Style_speech/LibriTTS/wav1600/2427'
# wav_fname '2427_154697_000016_000000.wav'
def write_single(output_folder, wav_fname, resample_rate, top_db=None):
    data, sample_rate = librosa.load(wav_fname, sr=None)
    # trim audio
    if top_db is not None:
        trimmed, _ = librosa.effects.trim(data, top_db=top_db)
    else:
        trimmed = data
    # resample audio
    resampled = librosa.resample(trimmed, sample_rate, resample_rate)
    y = (resampled * 32767.0).astype(np.int16)
    # wav_fname = wav_fname.split('/')[-1]  # 原来的

    # target_wav_fname = os.path.join(output_folder, wav_fname)  # 原来的
    target_wav_fname = os.path.join(output_folder, '{}-{}'.format(wav_fname.split('/')[-2], wav_fname.split('/')[-1]))  # CHANGE
    # target_txt_fname = os.path.join(output_folder, wav_fname.replace('.wav', '.txt'))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    write(target_wav_fname, resample_rate, y)
    # with open(target_txt_fname, 'wt') as f:
    #     f.write(text)
    #     f.close()


    """GRID加载文本"""
    read_txt = os.path.join("/data/conggaoxiang/GRID_dataset/alignments",wav_fname.split('/')[-2], '{}.align'.format(wav_fname.split('/')[-1].split('.wav')[0]))
    target_txt_fname = os.path.join('{}.txt'.format(target_wav_fname.split('.wav')[0]))

    if os.path.exists(read_txt):
        with open(read_txt, "r") as f:  # 打开文件
            data = f.read()  # 读取文件
            # print(data)
            read = data.split('\n')[1:-2]
            word_content = []
            a = 0
            for i in read:
                a = a + 1
                word = i.split(' ')[-1]
                word_content.append(word)
                if a < len(read):
                    word_content.append(' ')

            with open(target_txt_fname, "w") as f:
                # f.write(word_content)
                f.writelines(word_content)
                f.close()
    else:
        print(read_txt)








    return y.shape[0] / float(resample_rate)


def prepare_align_and_resample(data_dir, sr):
    wav_foder_names = ['audio_25k']
    wavs = []
    for wav_folder in wav_foder_names:
        wav_folder = os.path.join(data_dir, wav_folder)
        wav_fname_list = [str(f) for f in list(Path(wav_folder).rglob('*.wav'))]

        output_wavs_folder_name = 'wav{}'.format(sr//10)
        output_wavs_folder = os.path.join(data_dir, output_wavs_folder_name)
        if not os.path.exists(output_wavs_folder):
            os.mkdir(output_wavs_folder)

        for wav_fname in wav_fname_list:
            _sid = wav_fname.split('/')[-2]  ####  change!!!
            output_folder = os.path.join(output_wavs_folder, _sid)
            # txt_fname = wav_fname.replace('.wav','.normalized.txt')
            # with open(txt_fname, 'r') as f:
            #     text = f.readline().strip()
            # text = _clean_text(text, ['english_cleaners'])
            wavs.append((output_folder, wav_fname))

    lengths = Parallel(n_jobs=10, verbose=1)(
        delayed(write_single)(wav[0], wav[1], sr) for wav in wavs
    )
