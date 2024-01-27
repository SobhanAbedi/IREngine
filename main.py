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
    postings: Dict[int, PositionalPosting]


def nmz(cnt: str) -> str:
    return normalizer.normalize(cnt)


def remove_puncs(words: List[str]) -> List[str]:
    # TODO: remove punctuation words
    return words


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
        unpunc_words = remove_puncs(words)
        if not path.exists("dict_" + filename):
            pass
        uncommon_words = remove_common_words(unpunc_words)
        final_words = stem_list(uncommon_words)

        # Generate Positional Inverted Index:
        pii: Dict[str, WordEntry] = {}
        word_count = len(final_words)
        for word_idx in range(word_count):
            word = final_words[word_idx]
            if word in pii:
                entry = pii[word]
                entry.count += 1
                if doc_idx in entry.postings:
                    posting = entry.postings[doc_idx]
                    posting.count += 1
                    posting.positions.append(word_idx)
                else:
                    entry.postings[doc_idx] = PositionalPosting(1, [word_idx])
            else:
                pii[word] = WordEntry(1, {doc_idx: PositionalPosting(1, [word_idx])})

        for key, value in pii.items():
            print(f"{key}: {value.postings[0].positions}")
        break

    f.close()


if __name__ == '__main__':
    normalizer = h.Normalizer()
    stemmer = h.Stemmer()
    lemmatizer = h.Lemmatizer()
    main("data_500.json")
