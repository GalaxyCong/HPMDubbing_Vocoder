# HiFiGAN_16KHz
A 16kHz implementation of HiFi-GAN on LJ-Speech dataset.


## Detail setting




## Training
1. Please run
    ```
    python train_hifigan_16KHz.py --config config_v1_hifigan_16.json
    ```
    
## Inference
1. inference.py : wav -> mel -> wav
    ```
    python inference.py --checkpoint_file /data/conggaoxiang/vocoder/hifi-gan-master/My16_test1_hifigan/g_HiFi16
    ```
2. inference_e2e.py :  mel -> wav
    ```
    python inference_e2e.py --checkpoint_file /data/conggaoxiang/vocoder/hifi-gan-master/My16_test1_hifigan/g_HiFi16
    ```
    

## tensorboard
    ```
    tensorboard --logdir My16_test1_hifigan/logs/ --port=6005
    ```
