# Inspired by https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant/

import os
import struct
import torch
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

import pyaudio
import threading
import time
import numpy as np

torch.random.manual_seed(0)
device = torch.device('cpu')

print(f"PyTorch Version: {torch.__version__}, Pytorchaudio Version: {torchaudio.__version__}, \
Targeted Device: {device}")

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

class Listener:
    '''
    sr - sample rate
    chunk - number of samples (chunks) provided directly from audio card. - typ. 1024
    dt - data type of samples - typ. pyaudio.paInt8 (audio cards tend to be 8-bit)
    nchan - number of channels of audio to recordd. - typ. 1
    '''
    def __init__(self, sr, dt, chunk, nchan):
        self.chunk_size = chunk
        self.sample_rate = sr
        self.sample_datatype = dt
        self.nchannels = nchan
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.sample_datatype,
                                  channels=self.nchannels,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk_size)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("I am now listening... \n")


class SpeechRecognitionEngine:
    '''
    sr - sample rate - typ. 8000
    chunk - number of samples (chunks) provided directly from audio card. - typ. 1024
    dt - data type of samples - typ. pyaudio.paInt8 (audio cards tend to be 8-bit)
    nchan - number of channels of audio to recordd. - typ. 1
    '''
    def __init__(self, sr, dt, chunk, nchan, lookback, filepath=''):
        self.listener = Listener(sr=sr, dt=dt, chunk=chunk, nchan=nchan)
        self.audio_q = list()
        self.sample_rate = sr
        self.sample_data_type = dt
        self.nchannels = nchan
        self.filepath = filepath

    def run(self, qdepth):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop, args=(qdepth,), daemon=True)
        thread.start()

    def inference_loop(self, qdepth):
        i = 0
        zeros = np.zeros((CHUNK_SIZE,), dtype=int)
        pred_q = [struct.pack(f'{CHUNK_SIZE}h', *zeros)] * LOOKBACK
        while True:
            if len(self.audio_q) < qdepth:
                continue
            else:
                pred_q.extend(self.audio_q)
                pred_q = pred_q[QUEUE_DEPTH:]

                self.audio_q.clear()

                start = time.time()
                samples = [sample for s in pred_q for sample in struct.unpack(f'{CHUNK_SIZE}h', s)]
                waveform = torch.Tensor(samples).reshape(1, len(samples))
                waveform = waveform.to(device)

                with torch.inference_mode():
                    emission, _ = model(waveform)
                    beam_search_result = beam_search_decoder(emission)
                finish = time.time()
                beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
                print(f"Transcript: {beam_search_transcript}")
                print(f"Time to perform inference (with decoding): {finish - start:.3f} seconds.")

                #self.save(pred_q, f'{self.filepath}/test{str(i)}.wav')
                i += 1

            # time.sleep(0.05)


if __name__ == '__main__':
    SAMPLE_DATATYPE = pyaudio.paInt16
    SAMPLE_RATE = 16000 # data type of model
    CHUNK_SIZE = 1024
    QUEUE_DEPTH = 8 #as written, 0.5 seconds of data makes a big difference in performance
    LOOKBACK = QUEUE_DEPTH * 6 # 3s of lookback
    NCHANNELS = 1
    window_size = CHUNK_SIZE * QUEUE_DEPTH / SAMPLE_RATE
    print(f'Sample from audio card and save off data every {window_size}s')

    try:
        asr_engine = SpeechRecognitionEngine(SAMPLE_RATE, SAMPLE_DATATYPE, CHUNK_SIZE, NCHANNELS, LOOKBACK)
        asr_engine.run(QUEUE_DEPTH)
        threading.Event().wait()

    except KeyboardInterrupt as e:
        print('Exiting...')
        exit()
