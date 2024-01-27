# Important Note: This code is implemented very dangerously and should be edited carefully

import ujson
import hazm as h
from typing import List, Dict
from os import path
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import heapq
import time


@dataclass_json
@dataclass
class PositionalPosting:
    count: int
    positions: List[int]


@dataclass_json
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


def get_clean_words(content: str, common_words_filename: str) -> List[str]:
    # Normalize content
    nmz_cnt = nmz(content)
    words = h.word_tokenize(nmz_cnt)
    unpunc_words = remove_puncs(words)
    if not path.exists(common_words_filename):
        pass
    uncommon_words = remove_common_words(unpunc_words)
    return stem_list(uncommon_words)


def add_doc_words_to_pii(pii: Dict[str, WordEntry], words: List[str], doc_idx: int) -> None:
    word_count = len(words)
    for word_idx in range(word_count):
        word = words[word_idx]
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


def save_and_remove_common_words(pii:Dict[str, WordEntry], count:int, common_words_filename:str) -> None:
    # Find Common Words
    pii_list = list(pii.items())
    most_common = heapq.nlargest(count, pii_list, key=lambda x: x[1].count)
    most_common_words = [''] * count
    for i in range(len(most_common)):
        most_common_words[i] = most_common[i][0]
    # Save them
    with open(common_words_filename, "w") as f:
        ujson.dump(most_common_words, f)
    # Remove them form pii
    for word in most_common_words:
        del pii[word]


def save_dict(pii: Dict[str, WordEntry], dict_filename:str) -> None:
    words = sorted(pii.keys())
    with open(dict_filename, "w") as f:
        ujson.dump(words, f)


def load_pii(pii_filename: str) -> Dict[str, WordEntry]:
    pii: Dict[str, WordEntry] = {}
    with open(pii_filename, "r") as f:
        json_data = ujson.load(f)
        for k, v in json_data.items():
            pii[k] = WordEntry(**v)
    return pii


def save_pii(pii: Dict[str, WordEntry], pii_filename: str) -> None:
    with open(pii_filename, "w") as f:
        ujson.dump(pii, f, default=lambda x: x.to_dict())


def main(filename: str):
    f = open(filename, "r")
    filename_noformat = filename.split(".")[0]
    dict_filename = filename_noformat + "_dict.json"
    common_words_filename = filename_noformat + "_common.json"
    pii_filename = filename_noformat + "_pii.json"
    data = ujson.load(f)

    pii: Dict[str, WordEntry] = {}
    for key, value in data.items():
        # Preprocessing:
        words = get_clean_words(value['content'], common_words_filename)
        # Add to Positional Inverted Index:
        add_doc_words_to_pii(pii, words, int(key))

    # If Common Words file doesn't exist it also means that they haven't been removed during preprocessing
    if not path.exists(common_words_filename):
        save_and_remove_common_words(pii, 50, common_words_filename)

    # Save the Positional Inverted Index
    if not path.exists(dict_filename):
        save_dict(pii, dict_filename)

    # Save the Positional Inverted Index
    if not path.exists(pii_filename):
        save_pii(pii, pii_filename)

    f.close()


if __name__ == '__main__':
    normalizer = h.Normalizer()
    stemmer = h.Stemmer()
    lemmatizer = h.Lemmatizer()
    main("data_50.json")
