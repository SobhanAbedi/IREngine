# Important Note: This code is implemented very dangerously and should be edited carefully

import ujson
import hazm as h
from typing import List, Dict
from os import path
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import heapq
from math import sqrt
import time


@dataclass_json
@dataclass
class PositionalPosting:
    count: int
    positions: List[int]


@dataclass_json
@dataclass
class WordEntry:
    collection_count: int
    document_count: int
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
            entry.collection_count += 1
            if doc_idx in entry.postings:
                posting = entry.postings[doc_idx]
                posting.count += 1
                posting.positions.append(word_idx)
            else:
                entry.postings[doc_idx] = PositionalPosting(1, [word_idx])
                entry.document_count += 1
        else:
            pii[word] = WordEntry(1, 1, {doc_idx: PositionalPosting(1, [word_idx])})


def save_and_remove_common_words(pii:Dict[str, WordEntry], count:int, common_words_filename:str) -> None:
    # Find Common Words
    pii_list = list(pii.items())
    most_common = heapq.nlargest(count, pii_list, key=lambda x: x[1].collection_count)
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


def gen_stats(pii: Dict[str, WordEntry], doc_count: int, stats_filename: str) -> List[int]:
    doc_norm_facts: List[int] = [0] * doc_count
    for key, we in pii.items():
        for doc_idx, posting in we.postings.items():
            doc_norm_facts[doc_idx] += posting.count * posting.count
    for i in range(doc_count):
        doc_norm_facts[i] = round(sqrt(doc_norm_facts[i]))
    stats = {"N": doc_count, "dSize": doc_norm_facts}
    with open(stats_filename, 'w') as f:
        ujson.dump(stats, f)
    return doc_norm_facts


def gen_champ_list(pii: Dict[str, WordEntry], champ_list_size: int, champ_filename: str) -> Dict[str, WordEntry]:
    pii_champs: Dict[str, WordEntry] = {}
    for word, we in pii.items():
        postings_list = list(we.postings.items())
        champ_postings_list = heapq.nsmallest(champ_list_size,
                                              heapq.nlargest(champ_list_size, postings_list, key=lambda x: x[1].count),
                                              key=lambda x: x[0])
        champ_postings: Dict[int, PositionalPosting] = {}
        for posting in champ_postings_list:
            champ_postings[posting[0]] = posting[1]
        pii_champs[word] = WordEntry(we.collection_count, we.document_count, champ_postings)
    with open(champ_filename, "w") as f:
        ujson.dump(pii_champs, f, default=lambda x: x.to_dict())
    return pii_champs


def load_stats(stats_filename: str) -> (int, List[int]):
    doc_count: int
    doc_norm_facts: List[int]
    with open(stats_filename, 'r') as f:
        json_data = ujson.load(f)
        doc_count = json_data['N']
        doc_norm_facts = json_data['dSize']
    return doc_count, doc_norm_facts


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
    filename_noformat = filename.split(".")[0]
    dict_filename = filename_noformat + "_dict.json"
    common_words_filename = filename_noformat + "_common.json"
    pii_filename = filename_noformat + "_pii.json"
    pii_champs_filename = filename_noformat + "_pii_champs.json"
    stats_filename = filename_noformat + "_stats.json"

    f = open(filename, "r")
    data = ujson.load(f)

    doc_count = 0
    pii: Dict[str, WordEntry] = {}
    for key, value in data.items():
        # Preprocessing:
        words = get_clean_words(value['content'], common_words_filename)
        # Add to Positional Inverted Index:
        add_doc_words_to_pii(pii, words, int(key))
        doc_count += 1

    # If Common Words file doesn't exist it also means that they haven't been removed during preprocessing
    if not path.exists(common_words_filename):
        save_and_remove_common_words(pii, 50, common_words_filename)

    # Save the Positional Inverted Index
    if not path.exists(dict_filename):
        save_dict(pii, dict_filename)

    # Save the Positional Inverted Index
    if not path.exists(pii_filename):
        save_pii(pii, pii_filename)

    # Calculate Doc Normalization Factors
    if not path.exists(stats_filename):
        gen_stats(pii, doc_count, stats_filename)

    # Generate and save PII Champ list
    if not path.exists(pii_champs_filename):
        gen_champ_list(pii, 2, pii_champs_filename)

    f.close()


if __name__ == '__main__':
    normalizer = h.Normalizer()
    stemmer = h.Stemmer()
    lemmatizer = h.Lemmatizer()
    main("data_50.json")
