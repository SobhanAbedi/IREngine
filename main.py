import ujson as json
import hazm as h
from typing import List, Dict
from os import path
from dataclasses import dataclass

@dataclass
class PositionalPosting:
    count: int
    positions: List[int]


@dataclass
class WordEntry:
    count: int
    postings: List[PositionalPosting]


def nmz(cnt: str) -> str:
    return normalizer.normalize(cnt)


def remove_common_words(words: List[str]) -> List[str]:
    # TODO: remove common words
    return words


def stem_list(words: List[str]) -> List[str]:
    word_count = len(words)
    stemmed_words = [''] * word_count
    for i in range(word_count):
        stemmed_words[i] = stemmer.stem(words[i])
    return stemmed_words


def main(filename: str):
    f = open(filename, "r")
    data = json.load(f)

    for key, value in data.items():
        doc_idx = int(key)
        # Preprocessing:
        raw_cnt = value['content']
        # Normalize content
        nmz_cnt = nmz(raw_cnt)
        words = h.word_tokenize(nmz_cnt)
        if not path.exists("dict_" + filename):
            pass
        uncommon_words = remove_common_words(words)
        final_words = stem_list(uncommon_words)

        # Generate Positional Inverted Index:

    f.close()


if __name__ == '__main__':
    normalizer = h.Normalizer()
    stemmer = h.Stemmer()
    lemmatizer = h.Lemmatizer()
    main("data_500.json")
