import base64
import json

from google.cloud import datastore


project_id = 'speech-similarity'
datastore_client = datastore.Client(project_id)


def compare_voices(event, context):
    if event.get('data'):
        message_data = base64.b64decode(event['data']).decode('utf-8')
        message = json.loads(message_data)
    else:
        raise ValueError('Data sector is missing in the Pub/Sub message.')

    print(message)

    dic = dict(message)
    dic['score'] = .5

    with datastore_client.transaction():
        incomplete_key = datastore_client.key('Statistics')
        user = datastore.Entity(key=incomplete_key)
        user.update(dic)
        datastore_client.put(user)

