#!/usr/bin/env python
import os
import string
import random
import re
import pickle
# import autocomplete
# from autocomplete import models
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter
from nltk.util import ngrams

def load_training_data():
    unigrams = Counter()
    bigrams = []
    for file in os.listdir('guttenberg/'):
        with open('guttenberg/' + file, 'r') as f:
            content = f.read()
            content = re.sub('[^A-Za-z\']', " ", content)
            content = re.sub("\s\s+" , " ", content)
            content = content.replace(" '", "'")
            content = content.strip()
            content = content.split()
            unigrams = unigrams + Counter(ngrams(content, 1))
            bigrams = bigrams + list(ngrams(content, 2))
    return unigrams, bigrams

unigrams, bigrams = load_training_data()
bigrams_map = {word1:Counter() for word1, word2 in bigrams}
for bigram in bigrams:
    bigrams_map[bigram[0]].update([bigram[1]])

def get_suggestions(s1, s2, bigrams_map):
    suggestions = {word:freq for word, freq in bigrams_map[s1].items() if w.startswith(s2)}
    return Counter(suggestions).most_common(10)


def sort_tuple(tup): 
    tup.sort(key = lambda x: x[1], reverse = True)  
    return tup  

def generate_possibilities(current_string):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    suggestions = []
    for letter in alphabet:
        generate_output = generate_all(current_string, letter) 
        for more_output in generate_output:
            if more_output not in suggestions:
                suggestions.append(more_output)
    return sort_tuple(suggestions)[:3]

def generate_all(current_string, next_char):
    top_tup_words = []
    seen_characters = set()
    ## Replace with our own
    words = get_suggestions(current_string, next_char)
    for tup_word in words:
        if len(top_tup_words) == 3:
            break
        elif tup_word[0][len(next_char)] not in seen_characters:
            seen_characters.add(tup_word[0][len(next_char)])
            top_tup_words.append(tup_word)
    return top_tup_words # [(w, s). ....]

def generate_word(s2):
    seen_characters = set()
    top_tup_words = []
    words = [(k, v) for k, v in unigrams.most_common() if k.startswith(s2) and k != s2]
    for tup_word in words:
        if len(top_tup_words) == 3:
            break
        elif tup_word[0][len(s2)] not in seen_characters:
            seen_characters.add(tup_word[0][len(s2)])
            top_tup_words.append(tup_word)
    return top_tup_words

def get_characters(list, s2):
    output_list = []
    for word in list:
        output_list.append(word[0][len(s2)])
    return output_list
  
def predict(s1, s2):
    if len(s1) == 0:
        suggestions = generate_word(s2)
    elif len(s2) == 0:
        suggestions = generate_possibilities(s1)
    else:
        suggestions = generate_all(s1, s2)

    # Predict without any history
    if len(suggestions) < 3:
        new_suggestions = generate_word(s2)
        for i in range(3):
            if len(new_suggestions) - 1 < i:
                suggestions.append((s2 + "s", 0))
            elif new_suggestions[i][0] not in [j[0] for j in suggestions]:
                suggestions.append(new_suggestions[i])
            
            if len(suggestions) == 3:
                break

    return get_characters(suggestions, s2)

def run_pred(data):
    preds = []
    all_chars = string.ascii_letters
    for inp in data:
        # this model just predicts a random character each time
        words = inp.split()
        if len(words) == 1:
            prediction = self.predict('', words[0])
        else:
            prediction = self.predict(words[len(words) - 2], words[len(words) - 1])

        preds.append(''.join(prediction))

        '''top_guesses = [random.choice(all_chars) for _ in range(3)]
        preds.append(''.join(top_guesses))'''

    return preds