"""Some basic features for SVM (and maybe LSTM)"""
from collections import Counter
import numpy as np
import re
import libEscansion # parse metrical style en espanol

########## Linguistic features ##########
vowel = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
def get_metrical_vector(text):
    # one sonnet
    feature_vector = []
    for line in text:
        feature = scansion_parsing(line)
        feature_vector += feature
    return np.array(feature_vector)
    
def scansion_parsing(line):
    verse = libEscansion.VerseMetre(line)
    features = {
        'syllable_count': verse.count,
        'syllables': verse.syllables,
        'nuclei': verse.nuclei,
        'rhyme': verse.rhyme,
        'assonance': verse.asson,
        'rhythm_pattern': verse.rhythm,
        'rhythm_stressed_count': verse.rhythm.count('+'),  # Count stressed syllables
        'rhythm_unstressed_count': verse.rhythm.count('-'),  # Count unstressed syllables
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
    text = re.sub(r'[^a-zA-Z\s]', '', text)
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
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        unigrams = generate_unigrams(text)
        bigrams = generate_bigrams(text)
        unigram_counter.update(unigrams)
        bigram_counter.update(bigrams)
    
    # Get the top N unigrams and bigrams
    top_unigrams = set(word for word, _ in unigram_counter.most_common(n))
    top_bigrams = set(bigram for bigram, _ in bigram_counter.most_common(n))
    
    return top_unigrams, top_bigrams