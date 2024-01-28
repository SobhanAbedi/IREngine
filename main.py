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
import regex as r


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


class PreprocessorObjs:
    def __init__(self):
        self.email_reg = r.compile(r"\b[\w.-]+@\w+(\.\w+)+\b")
        self.number_reg = r.compile(r"\d+")
        self.space_reg = r.compile(r"[\s،\u200c]+")
        self.punctuation_reg = r.compile(r"[–()،؛»«:.!؟/;\u064B-\u0652\u0670\u0640']")
        self.suffixes = r.compile(r"\s+(ی|ای|ها|های|هایی|تر|تری|ترین|گر|گری|ام|ات|اش)\b")
        self.prefixes = r.compile(r"\b(می|نمی)\s+")
        self.stemmer = h.Stemmer()


def remove_sorted_list(original_list: List[str], sorted_list: List[str]) -> List[str]:
    new_list: List[str] = []
    for word in original_list:
        idx = bisect.bisect_left(sorted_list, word)
        if not (idx != len(sorted_list) and sorted_list[idx] == word):
            new_list.append(word)
    return new_list


def normalize(cnt: str) -> str:
    # Normalize Spaces. Should be done first to avoid errors due to misuse of space equivalents
    cnt = PPO.space_reg.sub(" ", cnt)

    # Remove from most common so that content gets smaller asap
    # Remove punctuations
    cnt = PPO.punctuation_reg.sub("", cnt)
    # Remove Numbers
    cnt = PPO.number_reg.sub("", cnt)
    # Remove emails
    cnt = PPO.email_reg.sub("", cnt)

    # Replace Arabic characters with their Persian equivalents
    cnt = cnt.replace("ي", "ی").replace("ك", "ک").replace("آ", "ا")
    # Replace 10 Spelling Variants. Replaced تومان with تومن because stemmer would probably mess that up
    cnt = cnt.replace("هيأت", "هيئت").replace("شترنج", "شطرنج")
    cnt = cnt.replace("ملیون", "میلیون").replace("تومان", "تومن")
    cnt = cnt.replace("اطاق", "اتاق").replace("ذغال", "زغال")
    cnt = cnt.replace("صواب", "ثواب").replace("دگمه", "دکمه")
    cnt = cnt.replace("طهران", "تهران").replace("طوفان", "توفان")
    # Replace space before suffixes with half-space
    half_space = "\u200c"
    cnt = PPO.suffixes.sub(half_space + r"\1", cnt)
    # Replace space after prefixes with half-space
    cnt = PPO.prefixes.sub(r"\1" + half_space, cnt)

    return cnt


def tokenize(cnt: str) -> List[str]:
    words: List[str] = []
    for word in cnt.split(' '):
        word_stem = PPO.stemmer.stem(word)
        if len(word_stem) > 0:
            words.append(word_stem)
    return words


def get_clean_words(content: str, common_words_filename: str) -> List[str]:
    # Normalize content
    cnt = normalize(content)
    # Tokenize, Stem and remove empty
    words = tokenize(cnt)

    if path.exists(common_words_filename):
        most_common_words: List[str]
        with open(common_words_filename, 'r') as f:
            most_common_words = ujson.load(f)
            print(f'Initial words: {words}')
            print(f'Most common words: {most_common_words}')
            words = remove_sorted_list(words, most_common_words)
            print(f'Resulted words: {words}')
    return words


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
                   stats_filename: str) -> (int, Dict[str, WordEntry]):
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
    return doc_count, pii


def daat_scoring(query_words: List[tuple[str, QWord]],
                 elimination_threshold: float, use_champs: bool) -> Dict[int, float]:
    idf_acc = 0.0
    scores: Dict[int, float] = {}
    for entry in query_words:
        if idf_acc > elimination_threshold:
            break
        idf_acc += entry[1].idf
        entry_dict = entry[1].champ_entry
        if not use_champs:
            entry_dict = entry[1].entry
        for doc_idx, posting in entry_dict.postings.items():
            if doc_idx in scores:
                scores[doc_idx] += entry[1].qf * entry[1].idf * posting.tf
            else:
                scores[doc_idx] = entry[1].qf * entry[1].idf * posting.tf
    return scores


def search_query(q_words: list[str], pii: Dict[str, WordEntry], pii_champs: Dict[str, WordEntry],
                 doc_count: int, filename: str) -> None:
    if len(q_words) == 0:
        # No valid words in query
        print('Query invalid!')
        return
    elif len(q_words) == 1:
        # One Valid word in query
        postings_list = list(pii_champs[q_words[0]].postings.items())
        sorted_query_words = heapq.nlargest(K, postings_list, key=lambda x: x[1].tf)
        print("Results:")
        with open(filename, "r") as f:
            data = ujson.load(f)
            for posting in sorted_query_words:
                print(data[str(posting[0])]['title'])
    else:
        # Multiple valid words in query
        # Set elimination threshold based on words in query
        base_elimination_threshold = 0.7
        indexed_elimination_threshold = {2: 0.95, 3: 0.90, 4: 0.85, 5: 0.8, 6: 0.75}
        elimination_threshold: float
        q_len = len(q_words)
        if q_len in indexed_elimination_threshold:
            elimination_threshold = indexed_elimination_threshold[q_len]
        else:
            elimination_threshold = base_elimination_threshold

        # Generate query dictionary
        q_dict: Dict[str, QWord] = {}
        for word in q_words:
            if word not in pii:
                print(f'{word} not in pii')
                continue
            if word in q_dict:
                q_dict[word].qf += 1.0
            else:
                entry = pii[word]
                q_dict[word] = QWord(word, log10(doc_count / entry.document_count), 1.0, entry, pii_champs[word])

        # Normalize query frequency and idf
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

        sorted_query_words = sorted(q_dict.items(), key=lambda x: x[1].idf, reverse=True)
        # Print qf idf for each search query
        for entry in sorted_query_words:
            print(f'{entry[0]} --> ({entry[1].qf}, {entry[1].idf})')
        # Calculate scores
        scores: Dict[int, float] = daat_scoring(sorted_query_words, elimination_threshold, True)
        if len(scores) < K:
            scores = daat_scoring(sorted_query_words, elimination_threshold, False)

        # Get top K using max-heap
        # sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sorted_results = heapq.nlargest(K, scores.items(), key=lambda x: x[1])
        with open(filename, "r") as f:
            data = ujson.load(f)
            for idx in range(len(sorted_results)):
                # print(f'{sorted_results[idx][0]} --> {sorted_results[idx][1]}')
                print(data[str(sorted_results[idx][0])]['title'])


def main(filename: str, queries: List[str]):
    filename_noformat = filename.split(".")[0]
    dict_filename = filename_noformat + "_dict.json"
    common_words_filename = filename_noformat + "_common.json"
    pii_filename = filename_noformat + "_pii.json"
    pii_champs_filename = filename_noformat + "_pii_champs.json"
    stats_filename = filename_noformat + "_stats.json"

    doc_count: int
    pii: Dict[str, WordEntry]
    pii_champs: Dict[str, WordEntry]
    stats: List[float]

    t0 = time.perf_counter()
    if not (path.exists(pii_filename) and path.exists(stats_filename)):
        # Generate and save PII list and stats
        print("Generating PII and Stats")
        doc_count, pii = generate_index(filename, dict_filename, common_words_filename,
                                                    pii_filename, stats_filename)
    else:
        print("Loading PII and Stats")
        doc_count, stats = load_stats(stats_filename)
        pii = load_pii(pii_filename)

    if not path.exists(pii_champs_filename):
        # Generate and save PII Champ list
        print("Generating PII Champions List")
        pii_champs = gen_champ_list(pii, CHAMP_LIST_SIZE, pii_champs_filename)
    else:
        print("Loading PII Champions List")
        pii_champs = load_pii(pii_champs_filename)
    t1 = time.perf_counter()
    delta = t1 - t0
    print(f"Index Loading/Generation took {delta} seconds")

    t0 = time.perf_counter()
    for q in queries:
        print(f'Answers for query {q}:')
        q_words = get_clean_words(q, common_words_filename)
        search_query(q_words, pii, pii_champs, doc_count, filename)
    t1 = time.perf_counter()
    delta = t1 - t0
    print(f"Query Processing took {delta} seconds")


if __name__ == '__main__':
    PPO = PreprocessorObjs()
    COMMON_WORD_COUNT = 50
    K = 5
    CHAMP_LIST_SIZE = 50
    t0 = time.perf_counter()
    main("IR_data_news_12k.json", ["جلسه علنی اعضای نمایندگان شورای انقلاب اسلامی"])
    t1 = time.perf_counter()
    delta = t1 - t0
    print(f"Your code took {delta} seconds to run")
