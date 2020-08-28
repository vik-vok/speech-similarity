import base64
import json
import random

from google.cloud import datastore
from google.cloud import bigquery


project_id = 'speech-similarity'
datastore_client = datastore.Client(project_id)
client = bigquery.Client()
table_id = 'speech-similarity.statistics.recorded_voices'


def compare_voices(event, context):
    if event.get('data'):
        message_data = base64.b64decode(event['data']).decode('utf-8')
        message = json.loads(message_data)
    else:
        raise ValueError('Data sector is missing in the Pub/Sub message.')

    print(message)

    dic = dict(message)
    dic['score'] = random.uniform(.5, .99)


    rows = [(dic['voiceId'], dic['parentId'], dic['userId'], dic['score'], '2016-05-19T10:38:47.046465')]
 

    
    table = client.get_table(table_id)
    errors = client.insert_rows(table, rows)

    if errors == []:
        print("New recorded voice added successfully")
    else:
        print(errors)

    # with datastore_client.transaction():
    #     incomplete_key = datastore_client.key('Statistics')
    #     user = datastore.Entity(key=incomplete_key)
    #     user.update(dic)
    #     datastore_client.put(user)

