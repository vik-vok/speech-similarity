import base64
import json
import random
import requests

from google.cloud import datastore
from google.cloud import bigquery
from google.cloud import storage

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from scipy import spatial

encoder = VoiceEncoder()

project_id = 'speech-similarity'
datastore_client = datastore.Client(project_id)
client = bigquery.Client()
storage_client = storage.Client()
table_id = 'speech-similarity.statistics.recorded_voices'

ORIGINAL_VOICE_URL = 'https://vikvok-anldg2io3q-ew.a.run.app/originalvoices/{}'


def get_embedding(path):
    fpath = Path(path)
    wav = preprocess_wav(fpath)
    return encoder.embed_utterance(wav)


def similarity(path1, path2):
    embed1 = get_embedding(path1)
    embed2 = get_embedding(path2)
    return spatial.distance.cosine(embed1, embed2)


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_uri_from_url(url):
    uri = 'gs://{}'
    path = url[find(url, '/')[-2]+1:]
    return uri.format(path)


def get_original_url(original_voice_id):
    voice_json = requests.get(ORIGINAL_VOICE_URL.format(original_voice_id)).json()
    return voice_json['path']


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

    with open('/tmp/original.wav') as objj:
        client.download_blob_to_file(original_voice_uri, objj)

    with open('/tmp/recorded.wav') as obj:
        client.download_blob_to_file(recorded_voice_uri, obj)

    score = random.uniform(.5, .99)
    rows = [(recorded_voice_id, original_voice_id, user_id, score, message['created'])]

    table = client.get_table(table_id)
    errors = client.insert_rows(table, rows)

    if errors == []:
        print("New recorded voice added successfully")
    else:
        print(errors)
