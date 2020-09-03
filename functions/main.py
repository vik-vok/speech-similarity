import base64
import json
import random
import requests

from google.cloud import bigquery
from google.cloud import storage
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/nikalosa/Desktop/MACS/Final Project/MicroServices/Speech-Similarity-e6a5d5620dac.json"

from scipy import spatial
import numpy
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa
from sklearn.decomposition import PCA

# from resemblyzer import VoiceEncoder, preprocess_wav
# from pathlib import Path
# from scipy import spatial

# encoder = VoiceEncoder()

project_id = 'speech-similarity'
b_client = bigquery.Client()
storage_client = storage.Client()
table_id = 'speech-similarity.statistics.recorded_voices'

ORIGINAL_VOICE_URL = 'https://vikvok-anldg2io3q-ew.a.run.app/originalvoices/{}'


# def get_embedding(path):
#     fpath = Path(path)
#     wav = preprocess_wav(fpath)
#     return encoder.embed_utterance(wav)


# def similarity(path1, path2):
#     embed1 = get_embedding(path1)
#     embed2 = get_embedding(path2)
#     return spatial.distance.cosine(embed1, embed2)


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_uri_from_url(url):
    uri = 'gs://{}'
    path = url[find(url, '/')[-2]+1:]
    return uri.format(path)


def get_original_url(original_voice_id):
    voice_json = requests.get(ORIGINAL_VOICE_URL.format(original_voice_id)).json()
    return voice_json['path']


def get_mfcc(path):
    signal, sample_rate = librosa.load(path)  # 8kHz
    signal = signal[:]  # Keep the first 3.5 seconds
    #     plt.plot(signal, c='b')
    # sd.play(signal)

    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])  # Perfect
    #     plt.plot(emphasized_signal, c='b')
    #     sd.play(emphasized_signal)

    frame_size = 0.025
    frame_stride = 0.01

    # Convert from seconds to samples
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate

    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z)

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0,
                                                                                                   num_frames * frame_step,
                                                                                                   frame_step),
                                                                                      (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    print(frames.shape)

    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation

    NFFT = 256  # 512 # or 256
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    num_ceps = 20
    cep_lifter = 22  # refers to the dimensionality of the MFCC vector in the original formulation.

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift

    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

    return mfcc


def mfcc_process(path):
    mfcc = get_mfcc(path)
    pca = PCA(n_components=12)
    pca.fit(mfcc.transpose())
    return pca.transform(mfcc.transpose()).transpose()


def compare_voices(event, context):
    if event.get('data'):
        message_data = base64.b64decode(event['data']).decode('utf-8')
        message = json.loads(message_data)
    else:
        raise ValueError('Data sector is missing in the Pub/Sub message.')

    message = dict(message)
    user_id = message['userId']
    original_voice_id = message['originalVoiceId']
    recorded_voice_id = message['recordedVoiceId']
    recorded_voice_uri = get_uri_from_url(message['voiceUrl'])

    original_voice_url = get_original_url(original_voice_id)
    original_voice_uri = get_uri_from_url(original_voice_url)

    print(original_voice_uri)
    print(recorded_voice_uri)

    with open('/tmp/original.wav', "wb") as objj:
        storage_client.download_blob_to_file(original_voice_uri, objj)

    with open('/tmp/recorded.wav', "wb") as obj:
        storage_client.download_blob_to_file(recorded_voice_uri, obj)

    mfcc_orig = mfcc_process('/tmp/original.wav')
    mfcc_rec = mfcc_process('/tmp/recorded.wav')

    score = spatial.distance.cosine(mfcc_orig.flatten(), mfcc_rec.flatten())

    if score > 1:
        score = 1

    score = min(0.5 + (1 - score)/2, random.uniform(.85, .99))

    rows = [(recorded_voice_id, original_voice_id, user_id, score, message['created'])]
    print(rows)
    table = b_client.get_table(table_id)
    errors = b_client.insert_rows(table, rows)

    if errors == []:
        print("New recorded voice added successfully")
    else:
        print(errors)
