from collections import defaultdict

# name list obtained from: https://www.ssa.gov/oact/babynames/decades/century.html
# accessed on Nov 6th, 2018

class PronounLexicon():
    def __init__(self, lexfile='pronouns.tsv'):
        self.lexicon = defaultdict(lambda : [])
        with open(lexfile) as fin:
            for line in fin:
                if len(line) > 2:
                    word = line.split()[0]
                    feats = dict(x.split('=') for x in line.split()[1].split(','))
                    for feat,val in feats.items():
                        self.lexicon['='.join([feat,val])].append(word)
        print(f"Read lexicon from {lexfile}:\n{self.lexicon}")

    def make_lex(self,feature,dictionary):
        '''
        given a fairseq dictionary, export a list of word idxs that match a desired feature
        '''
        return [idx for word,idx in dictionary.indices.items() if word.lower() in self.lexicon[feature]]

    def all_word_idxs(self,dictionary):
        return [idx for word,idx in dictionary.indices.items() if word.lower() in self.all_words()]

    def all_words(self):
        output = set()
        for subset in self.lexicon.values():
            for word in subset:
                output.add(word)
        return output
    
    def get_feature_set(self, feature_set):
        output = set()
        for t in feature_set:
            output |= set(self.lexicon[t])
        return output
    
    def annotate_feature_chunk_end(self, sentence, chunk_tags, feature_set):
        pronoun_lexicons = self.get_feature_set(feature_set)
        assert len(sentence) == len(chunk_tags)
        output = [0 for _ in range(len(sentence))]
        for i, (token, chunk_tag) in enumerate(zip(sentence, chunk_tags)):
            if token.lower() in pronoun_lexicons:
                if chunk_tag == 'O' or chunk_tag[:2] == 'U-':
                    output[i] = 1
                else:
                    chunk_type = chunk_tag[2:]
                    for j in range(i, len(sentence)):
                        end_chunk = chunk_tags[j]
                        assert end_chunk[2:] == chunk_type
                        if end_chunk[:2] == 'L-':
                            output[j] = 1
                            break
        return output
    
def find_gaps(sentence):
    gaps = []
    prev, cur = -1, -1
    while cur < len(marked_sentence):
        if sentence[cur] == 1:
            if prev != -1:
                gaps.append(cur - prev)
            prev = cur
        cur += 1
    return gaps

if __name__ == '__main__':
    lex = PronounLexicon()
    all_words = lex.all_words()
    in_file_path = "data/CBTest/data/cbt_train.txt"
    all_lens = []
    all_gaps = []
    with open(in_file_path) as f:
        for line in f:
            line = line.strip()
            marked_sentence = [1 if w in all_words else 0 for w in line.split(' ')]
            all_lens.append(len(marked_sentence))
            # print(marked_sentence)
            gaps = find_gaps(marked_sentence)
            # print(gaps)
            all_gaps.extend(gaps)
    import numpy as np
    print(np.mean(all_lens), np.std(all_lens))
    print(np.mean(all_gaps), np.std(all_gaps))

    # l = 32 covers 81.5% of the sentences
    # l = 64 covers 98.4% of the sentences
    l = 64
    print(len(list(filter(lambda x: x <= l, all_lens))) / float(len(all_lens)))

    # l = 10 covers 82.7% of the gaps
    # l = 20 covers 97.2% of the gaps
    # l = 30 covers 99.4% of the gaps
    l = 20
    print(len(list(filter(lambda x: x <= l, all_gaps))) / float(len(all_gaps)))
    
            

                
