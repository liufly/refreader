import sys
from collections import Counter

def read_file(in_file_path):
    sentences = []
    with open(in_file_path) as f:
        for line in f:
            line = line.strip()
            sentences.append(line.split(' '))
    return sentences

def write_dict(cnt, out_file_path):
    with open(out_file_path, 'w+') as f:
        for k, v in cnt.items():
            f.write(f"{k} {v}\n")

if __name__ == "__main__":
    all_sentences = []
    for in_file_path in sys.argv[1:-1]:
        all_sentences.extend(read_file(in_file_path))
    
    cnt = Counter([word for sentence in all_sentences for word in sentence])

    out_file_path = sys.argv[-1]
    write_dict(cnt, out_file_path)

    