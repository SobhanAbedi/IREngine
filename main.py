# Important Note: This code is implemented very dangerously and should be edited carefully

import ujson
import hazm as h
from typing import List, Dict
from os import path
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import heapq
from math import sqrt, log10
import time
import bisect


@dataclass_json
@dataclass
class PositionalPosting:
    count: int
    tf: float
    positions: List[int]


@dataclass_json
@dataclass
class WordEntry:
    collection_count: int
    document_count: int
    postings: Dict[int, PositionalPosting]


@dataclass_json
@dataclass
class QWord:
    word: str
    idf: float
    qf: float  # Query Freq
    champ_entry: WordEntry
    entry: WordEntry


def remove_sorted_list(original_list: List[str], sorted_list: List[str]) -> List[str]:
    new_list: List[str] = []
    for word in original_list:
        idx = bisect.bisect_left(sorted_list, word)
        if not (idx != len(sorted_list) and sorted_list[idx] == word):
            new_list.append(word)
    return new_list


def nmz(cnt: str) -> str:
    return normalizer.normalize(cnt)


def remove_puncs(words: List[str]) -> List[str]:
    # TODO: remove punctuation words
    puncs = sorted(['', '.', '!', ')', '(', '<', '>', ',', '/', ":"])
    return remove_sorted_list(words, puncs)


def stem_list(words: List[str]) -> List[str]:
    word_count = len(words)
    stemmed_words = [''] * word_count
    for i in range(word_count):
        stemmed_words[i] = stemmer.stem(words[i])
    return stemmed_words


def get_clean_words(content: str, common_words_filename: str) -> List[str]:
    nmz_cnt = nmz(content)
    words = h.word_tokenize(nmz_cnt)
    words = remove_puncs(words)
    if path.exists(common_words_filename):
        most_common_words: List[str]
        with open(common_words_filename, 'r') as f:
            most_common_words = ujson.load(f)
            words = remove_sorted_list(words, most_common_words)
    return stem_list(words)


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
                entry.postings[doc_idx] = PositionalPosting(1, 0.0, [word_idx])
                entry.document_count += 1
        else:
            pii[word] = WordEntry(1, 1, {doc_idx: PositionalPosting(1, 0.0, [word_idx])})


def save_and_remove_common_words(pii: Dict[str, WordEntry], count: int, common_words_filename: str) -> None:
    # Find Common Words
    pii_list = list(pii.items())
    most_common = heapq.nsmallest(count,
                                  heapq.nlargest(count, pii_list, key=lambda x: x[1].collection_count),
                                  key=lambda x: x[0])
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


def gen_tfs(pii: Dict[str, WordEntry]) -> None:
    for word, we in pii.items():
        for doc_idx, posting in we.postings.items():
            posting.tf = 1 + log10(posting.count)


def gen_stats(pii: Dict[str, WordEntry], doc_count: int, stats_filename: str) -> List[float]:
    doc_norm_facts: List[float] = [0] * doc_count
    for _, we in pii.items():
        for doc_idx, posting in we.postings.items():
            doc_norm_facts[doc_idx] += posting.tf * posting.tf
    for i in range(doc_count):
        doc_norm_facts[i] = sqrt(doc_norm_facts[i])
    stats = {"N": doc_count, "dSize": doc_norm_facts}
    with open(stats_filename, 'w') as f:
        ujson.dump(stats, f)
    return doc_norm_facts


def normalize_tfs(pii: Dict[str, WordEntry], norm_facts: List[float]) -> None:
    for word, we in pii.items():
        for doc_idx, posting in we.postings.items():
            posting.tf /= norm_facts[doc_idx]


def gen_champ_list(pii: Dict[str, WordEntry], champ_list_size: int, champ_filename: str) -> Dict[str, WordEntry]:
    pii_champs: Dict[str, WordEntry] = {}
    for word, we in pii.items():
        postings_list = list(we.postings.items())
        champ_postings_list = heapq.nsmallest(champ_list_size,
                                              heapq.nlargest(champ_list_size, postings_list, key=lambda x: x[1].tf),
                                              key=lambda x: x[0])
        champ_postings: Dict[int, PositionalPosting] = {}
        for posting in champ_postings_list:
            champ_postings[posting[0]] = posting[1]
        pii_champs[word] = WordEntry(we.collection_count, we.document_count, champ_postings)
    with open(champ_filename, "w") as f:
        ujson.dump(pii_champs, f, default=lambda x: x.to_dict())
    return pii_champs


def load_stats(stats_filename: str) -> (int, List[float]):
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
            postings: Dict[int, PositionalPosting] = {}
            for doc_idx, p in v['postings'].items():
                postings[doc_idx] = PositionalPosting(**p)
            pii[k] = WordEntry(v['collection_count'], v['document_count'], postings)
    return pii


def save_pii(pii: Dict[str, WordEntry], pii_filename: str) -> None:
    with open(pii_filename, "w") as f:
        ujson.dump(pii, f, default=lambda x: x.to_dict())


def generate_index(filename: str,
                   dict_filename: str,
                   common_words_filename: str,
                   pii_filename: str,
                   pii_champs_filename: str,
                   stats_filename: str) -> (int, Dict[str, WordEntry], Dict[str, WordEntry]):
    doc_count = 0
    pii: Dict[str, WordEntry] = {}
    pii_champs: Dict[str, WordEntry]
    stats: List[float]
    with open(filename, 'r') as f:
        data = ujson.load(f)
        for key, value in data.items():
            # Preprocessing:
            words = get_clean_words(value['content'], common_words_filename)
            # Add to Positional Inverted Index:
            add_doc_words_to_pii(pii, words, int(key))
            doc_count += 1
        # If Common Words file doesn't exist it also means that they haven't been removed during preprocessing
        if not path.exists(common_words_filename):
            save_and_remove_common_words(pii, COMMON_WORD_COUNT, common_words_filename)
        # Save the Positional Inverted Index
        if not path.exists(dict_filename):
            save_dict(pii, dict_filename)
        # Calculate Doc Normalization Factors
        gen_tfs(pii)
        stats = gen_stats(pii, doc_count, stats_filename)
        normalize_tfs(pii, stats)
        # Save the Positional Inverted Index
        save_pii(pii, pii_filename)
        # Generate and save PII Champ list
        pii_champs = gen_champ_list(pii, CHAMP_LIST_SIZE, pii_champs_filename)
    return doc_count, pii, pii_champs


def main(filename: str, queries: List[str]):
    filename_noformat = filename.split(".")[0]
    dict_filename = filename_noformat + "_dict.json"
    common_words_filename = filename_noformat + "_common.json"
    pii_filename = filename_noformat + "_pii.json"
    pii_champs_filename = filename_noformat + "_pii_champs.json"
    stats_filename = filename_noformat + "_stats.json"

    with open(common_words_filename, "r") as f:
        common_words = ujson.load(f)
        for word in common_words:
            print(word)

    doc_count: int
    pii: Dict[str, WordEntry]
    pii_champs: Dict[str, WordEntry]
    stats: List[float]
    if not (path.exists(stats_filename) and path.exists(pii_filename) and path.exists(pii_champs_filename)):
        doc_count, pii, pii_champs = generate_index(filename, dict_filename, common_words_filename,
                                                    pii_filename, pii_champs_filename, stats_filename)
    else:
        doc_count, stats = load_stats(stats_filename)
        pii = load_pii(pii_filename)
        pii_champs = load_pii(pii_champs_filename)

    for q in queries:
        print(f'Answers for query:{q}')
        q_words = get_clean_words(q, common_words_filename)
        if len(q_words) == 0:
            print('Query invalid!')
            continue
        elif len(q_words) == 1:
            postings_list = list(pii_champs[q_words[0]].postings.items())
            sorted_query_words = heapq.nlargest(K, postings_list, key=lambda x: x[1].tf)
            print("Results:")
            with open(filename, "r") as f:
                data = ujson.load(f)
                for posting in sorted_query_words:
                    print(data[str(posting[0])]['title'])
        else:
            base_elimination_ratio = 0.7
            indexed_elimination_ratio = {2: 0.95, 3: 0.90, 4: 0.85, 5: 0.8, 6: 0.75}
            elimination_ratio: float
            q_len = len(q_words)
            if q_len in indexed_elimination_ratio:
                elimination_ratio = indexed_elimination_ratio[q_len]
            else:
                elimination_ratio = base_elimination_ratio
            q_dict: Dict[str, QWord] = {}
            for word in q_words:
                if word not in pii:
                    print(f'{word} not in pii')
                    continue
                if word in q_dict:
                    q_dict[word].qf += 1.0
                else:
                    entry = pii[word]
                    q_dict[word] = QWord(word, log10(doc_count/entry.document_count), 1.0, entry, pii_champs[word])

            qf_norm_fact = 0
            idf_norm_fact = 0
            for word, q_word in q_dict.items():
                q_word.qf = 1 + log10(q_word.qf)
                qf_norm_fact += q_word.qf * q_word.qf
                idf_norm_fact += q_word.idf
            qf_norm_fact = sqrt(qf_norm_fact)
            for word, q_word in q_dict.items():
                q_word.qf /= qf_norm_fact
                q_word.idf /= idf_norm_fact

            scores: Dict[int, float] = {}
            sorted_query_words = sorted(q_dict.items(), key=lambda x: x[1].idf, reverse=True)
            idf_acc: float = 0.0
            for entry in sorted_query_words:
                print(f'{entry[0]} --> ({entry[1].qf}, {entry[1].idf})')
            for entry in sorted_query_words:
                if idf_acc > elimination_ratio:
                    break
                idf_acc += entry[1].idf
                for doc_idx, posting in entry[1].champ_entry.postings.items():
                    if doc_idx in scores:
                        scores[doc_idx] += entry[1].qf * entry[1].idf * posting.tf
                    else:
                        scores[doc_idx] = entry[1].qf * entry[1].idf * posting.tf
            if len(scores) < K:
                scores = {}
                idf_acc = 0.0
                for entry in sorted_query_words:
                    if idf_acc > elimination_ratio:
                        break
                    idf_acc += entry[1].idf
                    for doc_idx, posting in entry[1].entry.postings.items():
                        if doc_idx in scores:
                            scores[doc_idx] += entry[1].qf * entry[1].idf * posting.tf
                        else:
                            scores[doc_idx] = entry[1].qf * entry[1].idf * posting.tf
            sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            with open(filename, "r") as f:
                data = ujson.load(f)
                for idx in range(min(K, len(sorted_results))):
                    # print(f'{sorted_results[idx][0]} --> {sorted_results[idx][1]}')
                    print(data[str(sorted_results[idx][0])]['title'])


if __name__ == '__main__':
    normalizer = h.Normalizer()
    stemmer = h.Stemmer()
    lemmatizer = h.Lemmatizer()
    COMMON_WORD_COUNT = 50
    K = 5
    CHAMP_LIST_SIZE = 20
    t0 = time.perf_counter()
    main("IR_data_news_12k.json", ["جلسه امروز اعضای شورای انقلاب اسلامی"])
    t1 = time.perf_counter()
    total = t1 - t0
    print(f"Your code took {total} seconds to run")
