# Inspired by https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant/
# Simulate audio buffer scheme with file.

import argparse
import wave
import struct
import time

import numpy as np
import torch
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files
device = torch.device('cpu')


files = download_pretrained_files("librispeech-4-gram")
LM_WEIGHT = 3.23
WORD_SCORE = -0.26

beam_search_decoder = ctc_decoder(
    lexicon=files.lexicon,
    tokens=files.tokens,
    lm=files.lm,
    nbest=3,
    beam_size=1500,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Testing running pyaudio simulation to support streaming audio from laptop.")
    parser.add_argument('--file', '-p', type=str, default=None, required=True,
                        help='absolute file location of file to read.  Must be a .wav.')
    args = parser.parse_args()

    filename = args.file
    SAMPLE_RATE = 16000 # data type of model
    CHUNKSIZE = 1024
    QUEUEDEPTH = 8 #as written, 0.5 seconds of data makes a big difference in performance
    LOOKBACK = QUEUEDEPTH * 6
    NCHANNELS = 1
    window_size = CHUNKSIZE * QUEUEDEPTH / SAMPLE_RATE
    print(f'Sample from audio card and save off data every {window_size}s')

    wf = wave.open(filename)
    audio_q = []
    chunk = np.zeros((CHUNKSIZE,),dtype=int)
    pred_q = [struct.pack(f'{CHUNKSIZE}h', *chunk)] * LOOKBACK
    print(f'Buffer Lookback: {LOOKBACK * CHUNKSIZE/ SAMPLE_RATE}s of audio.  ')
    i = 0
    while True:

        # filling from sound card
        for _ in range(QUEUEDEPTH):
            bits = wf.readframes(CHUNKSIZE)
            if len(bits) == 0:
                print(f'Read all bytes from {filename}')
                exit()
            #print(f'Pushing {len(bits)}B chunks into audio queue.')
            audio_q.append(bits)

        pred_q.extend(audio_q)
        pred_q = pred_q[QUEUEDEPTH:]

        audio_q.clear()

        start = time.time()
        samples = [sample for s in pred_q for sample in struct.unpack(f'{CHUNKSIZE}h', s)]
        waveform = torch.Tensor(samples).reshape(1, len(samples))
        waveform = waveform.to(device)

        with torch.inference_mode():
            emission, _ = model(waveform)
            beam_search_result = beam_search_decoder(emission)
        finish = time.time()
        beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
        print(f"Transcript {i}: {beam_search_transcript}")
        print(f"Time to perform inference (with decoding): {finish - start:.3f} seconds.")
        i += 1
