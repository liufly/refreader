import json
import sys
import numpy as np

class Bert_Reader:

    def __init__(self, in_bert_json_file_path):
        self.in_bert_json_file_path = in_bert_json_file_path

    def read(self):
        data = []
        with open(self.in_bert_json_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                bert_json = json.loads(line)
                tokens = [feature['token'] for feature in bert_json['features']]
                layers = []
                nb_layers = len(bert_json['features'][0]['layers'])
                for i in range(nb_layers):
                    cur_layer = []
                    for feature in bert_json['features']:
                        assert -(i+1) == feature['layers'][i]['index']
                        values = feature['layers'][i]['values']
                        values = np.array(values)
                        cur_layer.append(values)
                    assert len(cur_layer) == len(tokens)
                    layers.append(cur_layer)
                layers = np.array(layers)
                # [nb_layers (4), seq_len, emb_size]
                data.append((tokens, layers))
        return data

if __name__ == "__main__":
    in_bert_json_file_path = "data/gap-bert-bin/gap-train.bert.jsonl"
    reader = Bert_Reader(in_bert_json_file_path)
    bert_data = reader.read()