import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.hparams import HParam
from utils.utils import read_wav_np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def main(hp, args):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").cpu()
    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)
    mel_path = hp.data.mel_path

    # Create all folders
    os.makedirs(mel_path, exist_ok=True)
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        sr, wav = read_wav_np(wavpath)
        assert sr == hp.audio.sampling_rate, \
            "sample rate mismatch. expected %d, got %d at %s" % \
            (hp.audio.sampling_rate, sr, wavpath)
        
        if len(wav) < hp.audio.segment_length + hp.audio.pad_short:
            wav = np.pad(wav, (0, hp.audio.segment_length + hp.audio.pad_short - len(wav)), \
                    mode='constant', constant_values=0.0)

        wav = torch.from_numpy(wav)
        input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.cpu()
        with torch.no_grad():
            mel = model(input_values).logits[0].T.cpu()
            mel /= 12 # normalize to ~(-1;1)

        id = os.path.basename(wavpath).split(".")[0]
        np.save('{}/{}.npy'.format(mel_path, id), mel.numpy(), allow_pickle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help="root directory of wav files")
    args = parser.parse_args()
    hp = HParam(args.config)

    main(hp, args)