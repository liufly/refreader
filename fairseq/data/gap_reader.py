import csv
import sys
import random
import numpy as np
import string
import re

from collections import namedtuple

# Fieldnames used in the gold dataset .tsv file.
GOLD_FIELDNAMES = [
    'ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'A-coref', 'B',
    'B-offset', 'B-coref', 'URL'
]

# Fieldnames expected in system output .tsv files.
SYSTEM_FIELDNAMES = ['ID', 'A-coref', 'B-coref']

GAP_Record = namedtuple("GAP_Record", ["example_id", "text", "pronoun", 
                        "pronoun_offset_start", "pronoun_offset_end",
                        "a", "a_offset_start", "a_offset_end", "a_coref",
                        "b", "b_offset_start", "b_offset_end", "b_coref"])

class GAP_Reader:

    def __init__(self, in_tsv_file_path, is_gold=True):
        self.in_tsv_file_path = in_tsv_file_path
        self.is_gold = is_gold

    def read(self):
        data = []
        fieldnames = GOLD_FIELDNAMES if self.is_gold else SYSTEM_FIELDNAMES
        with open(self.in_tsv_file_path, 'r') as in_tsv_file:
            reader = csv.DictReader(in_tsv_file, fieldnames=fieldnames, delimiter='\t')
            if self.is_gold:
                next(reader, None)
            for i, row in enumerate(reader):
                example_id = row['ID']

                text, pronoun, pronoun_offset_start, pronoun_offset_end = None, None, None, None
                a, a_offset_start, a_offset_end = None, None, None
                b, b_offset_start, b_offset_end = None, None, None

                if self.is_gold:
                    text = row['Text']
                    pronoun = row['Pronoun']
                    pronoun_offset_start, pronoun_offset_end = row['Pronoun-offset'].split(":")
                    pronoun_offset_start = int(pronoun_offset_start)
                    pronoun_offset_end = int(pronoun_offset_end) - 1
                    assert pronoun_offset_start == pronoun_offset_end

                    a = row['A']
                    a_offset_start, a_offset_end = row['A-offset'].split(":")
                    a_offset_start = int(a_offset_start)
                    a_offset_end = int(a_offset_end) - 1 # -1 to be inclusive
                    assert a_offset_start <= a_offset_end
                    assert a == ' '.join(text.split(' ')[a_offset_start:a_offset_end+1])
                
                    b = row['B']
                    b_offset_start, b_offset_end = row['B-offset'].split(":")
                    b_offset_start = int(b_offset_start)
                    b_offset_end = int(b_offset_end) - 1 # -1 to be inclusive
                    assert a_offset_start <= a_offset_end
                    assert b == ' '.join(text.split(' ')[b_offset_start:b_offset_end+1])
                    
                    assert a_offset_start < b_offset_start
                    assert a_offset_end < b_offset_end
                
                assert row['A-coref'].upper() in ['TRUE', 'FALSE']
                a_coref = True if row['A-coref'].upper() == 'TRUE' else False
                assert row['B-coref'].upper() in ['TRUE', 'FALSE']
                b_coref = True if row['B-coref'].upper() == 'TRUE' else False

                # data.append((
                #     example_id, text,
                #     (pronoun, pronoun_offset_start, pronoun_offset_end),
                #     (a, a_offset_start, a_offset_end, a_coref),
                #     (b, b_offset_start, b_offset_end, b_coref),
                # ))
                data.append(GAP_Record(
                    example_id, text, 
                    pronoun, pronoun_offset_start, pronoun_offset_end,
                    a, a_offset_start, a_offset_end, a_coref,
                    b, b_offset_start, b_offset_end, b_coref
                ))
        return data

if __name__ == "__main__":
    gap_reader = GAP_Reader('data/gap/gap-test.tok.tsv')
    a_cnt, b_cnt = 0, 0
    for gid, text, (p, pos, poe), (a, aos, aoe, ac), (b, bos, boe, bc) in gap_reader.read():
        assert (ac and (not bc)) or ((not ac) and bc) or ((not ac) and (not bc))
        if ac:
            a_cnt += 1
        if bc:
            b_cnt += 1
    print(a_cnt, b_cnt)