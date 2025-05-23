"""Some basic features for SVM (and maybe LSTM)"""
from collections import Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import fonemas
from pyverse import Pyverse
import silabeador
import regex as re # better re lib
import libEscansion # parse metrical style en espanol


########## Linguistic features ##########


def extract_features(line):
    line = re.sub(r"[^\p{L}\p{N}\s']", '', line)
    line = re.sub(r'(?<=\p{L})(?=\p{N})', ' ', line)
    line = re.sub(r'(?<=\p{N})(?=\p{L})', ' ', line)
    # Create a Pyverse object to analyze the verse
    verse = Pyverse(line)

    num_syllables = verse.count

    # estimate stressed syllables using syllable positions (based on syllabified sentence)
    stress_vector = [word.accentuation for word in verse.sentence.word_objects]
    stress_vector += [0] * (14 - len(stress_vector))  # pad to fixed length 14
    # phoneme transcription using fonemas library
    t = fonemas.Transcription(line, mono=False, epenthesis=True, aspiration=False, rehash=False)
    phonetic = t.phonetics.syllables
    num_phonemes = len(phonetic)

    return {
        'num_syllables': num_syllables,
        'accent_type': verse.sentence.last_word.accentuation,
        'num_phonemes': num_phonemes,
        'stress_vector': stress_vector # only one used at the moment
    }

def features_for_sonnet(lines):
    lines = lines.split('\n')
    features = [extract_features(line) for line in lines if len(line.strip()) > 0]
    df = pd.DataFrame(features)

    # Average vector of stress positions
    if 'stress_vector' not in df.columns:
        print(lines)
    stress_matrix = pd.DataFrame(df['stress_vector'].tolist())
    avg_stress_vector = stress_matrix.mean().tolist()

    return {
        **{f'stress_pos_{i}': val for i, val in enumerate(avg_stress_vector)}
    }
# Attempt to use scansion approach, very heavy computationally, would take 60hrs sequentially
vowel = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
def get_metrical_vector(text):
    # one sonnet
    feature_vector = []
    for line in text:
        feature = scansion_parsing(line)
        feature_vector += feature
    return feature_vector
    
def scansion_parsing(line):
    verse = libEscansion.VerseMetre(line)
    features = {
        'syllable_count': verse.count,
        'syllables': verse.syllables,
        'nuclei': verse.nuclei,
        'rhyme': verse.rhyme,
        'assonance': verse.asson,
        'rhythm_pattern': verse.rhythm,
        'rhythm_stressed_count': verse.rhythm.count('+'),  # count stressed syllables
        'rhythm_unstressed_count': verse.rhythm.count('-'),  # count unstressed syllables
        'ends_in_rhyme': verse.rhyme == verse.asson or verse.rhyme.endswith(verse.asson),  # ending in rhyme
        'last_vowel': verse.nuclei[-1] if verse.nuclei else ''
    }
    return features_to_array(features)

def features_to_array(features):
    syllable_count = features['syllable_count']
    rhythm_stressed_count = features['rhythm_stressed_count']
    rhythm_unstressed_count = features['rhythm_unstressed_count']
    last_vowel = features['last_vowel']
    last_vowel_embed = vowel.get(last_vowel, len(vowel))

    # Encode rhythm pattern as one-hot or numeric encoding
    rhythm_pattern_encoded = [1 if c == '+' else 0 for c in features['rhythm_pattern']]
    rhythm_pattern_encoded = rhythm_pattern_encoded + [0] * (15 - len(rhythm_pattern_encoded)) if len(rhythm_pattern_encoded) < 15 else rhythm_pattern_encoded[:15]
    # Number of vowels in nuclei (optional, could use specific vowel count)
    nuclei_vowels_count = sum(1 for char in features['nuclei'] if char in 'aeiou')

    # Create a feature vector
    feature_vector = [
        syllable_count,
        rhythm_stressed_count,
        rhythm_unstressed_count,
        nuclei_vowels_count,
        last_vowel_embed,
    ] + rhythm_pattern_encoded

    return feature_vector

########## Bag of words features ##########
# Strips out punctuation, for now

def generate_unigrams(text):
    words = text.split()
    return words

def generate_bigrams(text):
    words = text.split()
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]  # Pair consecutive words
    return bigrams

def text_to_bag_of_words(text, unigram_vocab, bigram_vocab):
    text = re.sub(r"[^\p{L}\p{N}\s']", '', text)
    text = re.sub(r'(?<=\p{L})(?=\p{N})', ' ', text)
    text = re.sub(r'(?<=\p{N})(?=\p{L})', ' ', text)
    unigrams = generate_unigrams(text)
    bigrams = generate_bigrams(text)

    unigram_counts = Counter(unigrams)
    bigram_counts = Counter(bigrams)

    feature_vector = []
    for word in unigram_vocab:
        feature_vector.append(unigram_counts.get(word, 0))  # Count occurrences of each unigram

    for bigram in bigram_vocab:
        feature_vector.append(bigram_counts.get(bigram, 0))  # Count occurrences of each bigram

    return np.array(feature_vector)

def get_top_n_vocab(texts, n=100):
    unigram_counter = Counter()
    bigram_counter = Counter()
    
    for text in texts:
        text = re.sub(r'[^\p{L}\p{N}\s]', '', text)
        unigrams = generate_unigrams(text)
        bigrams = generate_bigrams(text)
        unigram_counter.update(unigrams)
        bigram_counter.update(bigrams)
    
    # Get the top N unigrams and bigrams
    top_unigrams = list(word for word, _ in unigram_counter.most_common(n))
    top_bigrams = list(bigram for bigram, _ in bigram_counter.most_common(n))
    
    return top_unigrams, top_bigrams


# implement tf-idf
def get_tfidf_vocab(texts, n=100):
    token_uni = []
    token_bi = []
    tf_uni = []
    tf_bi = []
    uni_vocab, bi_vocab = get_top_n_vocab(texts, n=n)
    for text in texts:
        # an entire poem
        text = re.sub(r'[^\p{L}\p{N}\s]', '', text.lower())
        uni = generate_unigrams(text)
        bi = generate_bigrams(text)
        unigram_counter = Counter(uni)
        bigram_counter = Counter(bi)
        token_uni.append(uni)
        token_bi.append(bi)
        
        tf_uni.append({term: unigram_counter[term] / unigram_counter.total() for term in uni_vocab})

        tf_bi.append({term: bigram_counter[term]/bigram_counter.total() for term in bi_vocab})

    num_docs = len(texts)
    idf_uni = {}
    idf_bi = {}
    for term in uni_vocab:
        num_contain_uni = sum(1 for doc in token_uni if term in doc)
        idf_uni[term] = math.log((num_docs + 1)/(num_contain_uni + 1)) + 1
    for term in bi_vocab:
        num_contain_bi = sum(1 for doc in token_bi if term in doc)
        idf_bi[term] = math.log((num_docs + 1)/(num_contain_bi + 1)) + 1

    tfidf_uni = []
    tfidf_bi = []
    for tf in tf_uni:
        tfidf = {}
        for term in uni_vocab:
            tfidf[term] = tf[term] * idf_uni[term]
        tfidf_uni.append(tfidf)
    for tf in tf_bi:
        tfidf = {}
        for term in bi_vocab:
            tfidf[term] = tf[term] * idf_bi[term]
        tfidf_bi.append(tfidf)  
    out_vect = [list(a.values()) + list(b.values()) for a, b, in zip(tfidf_uni, tfidf_bi)]
    return out_vect, uni_vocab, bi_vocab, tf_uni, tf_bi, idf_uni, idf_bi

def get_tfidf_test(text, uni_vocab, bi_vocab, tf_uni, tf_bi, idf_uni, idf_bi):
    text = re.sub(r'[^\p{L}\p{N}\s]', '', text.lower())
    uni = generate_unigrams(text)
    bi = generate_bigrams(text)
    unigram_counter = Counter(uni)
    bigram_counter = Counter(bi)

        
    tf_uni = {term: unigram_counter[term] / unigram_counter.total() for term in uni_vocab}

    tf_bi = {term: bigram_counter[term]/bigram_counter.total() for term in bi_vocab}
    tfidf_uni = {}
    tfidf_bi = {}
    for term in uni_vocab:
        tfidf_uni[term] = tf_uni[term] * idf_uni[term]
    for term in bi_vocab:
        tfidf_bi[term] = tf_bi[term] * idf_bi[term]
    return list(tfidf_uni.values()) + list(tfidf_bi.values())