import json
import sys

class witLoadFromJson(object):
    def __init__(self):
        with open('data/BotCycle/expressions.json') as expressions_file:
            self.data = json.load(expressions_file)

    def getSentencesWithIntent(self):
        array = self.data['data']
        result = list(map(lambda x: {'sentence': x['text'], 'intent': next( (ent for ent in x['entities'] if ent['entity'] == "intent"), {}).get('value', 'none').strip('"') }, array))
        print(result)
        return result