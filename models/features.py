from rantanplan.core import get_scansion # parse metrical style en espanol
import numpy as np

def get_metrical_vector(text):
    feature_vector = []
    for line in text:
        feature = scansion_parsing(line)
        feature_vector += feature
    return np.array(feature_vector)
    
def scansion_parsing(line):
    """
    Returns an array with:
        - percentage stressed syllables
        - last syllable stress count
        - existence of synalepha
        - percentage of stress in rhythm
        - total stressed in rhythm
        - total unstressed
    """
    out = get_scansion(line)
    total_syllables = 0
    stressed_syllables = 0
    word_end_stressed = 0
    synalepha_present = 0
    total_rhythm_stressed = 0
    total_rhythm_unstressed = 0
    
    rhythm_pattern = out['rhythm']['stress']
    total_rhythm_stressed = rhythm_pattern.count('+')
    total_rhythm_unstressed = rhythm_pattern.count('-')
    
    for token in out['tokens']:
        for syllable_data in token['tokens']:
            syllable = syllable_data['word'][0]
            is_stressed = syllable['is_stressed']
            is_word_end = syllable.get('is_word_end', False)
            has_synalepha = syllable.get('has_synalepha', False)
            
            total_syllables += 1
            stressed_syllables += 1 if is_stressed else 0
            word_end_stressed += 1 if is_word_end and is_stressed else 0
            synalepha_present = max(synalepha_present, 1 if has_synalepha else 0)
    
    stress_ratio = stressed_syllables / total_syllables if total_syllables > 0 else 0
    rhythm_ratio = total_rhythm_stressed / len(rhythm_pattern) if len(rhythm_pattern) > 0 else 0

    # Return a condensed feature vector for this line
    return [
        stress_ratio,
        word_end_stressed,
        synalepha_present,
        rhythm_ratio, 
        total_rhythm_stressed,
        total_rhythm_unstressed]