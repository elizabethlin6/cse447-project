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



class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def __init__(self, unigrams = Counter(), bigrams_map = {}):
        self.bigrams_map = bigrams_map
        self.unigrams = unigrams
    
    @classmethod
    def load_training_data(cls):
        unigrams = Counter()
        bigrams = []
        for file in os.listdir('data/multilang/'):
            with open('data/multilang/' + file, 'r') as f:
                content = f.read()
                content = re.sub('[^A-Za-z\']', " ", content)
                content = re.sub("\s\s+" , " ", content)
                content = content.replace(" '", "'")
                content = content.strip()
                content = content.split()
                unigrams = unigrams + Counter(content)
                bigrams = bigrams + list(ngrams(content, 2))
        return unigrams, bigrams

    @classmethod
    def generate_bigram_map(cls, bigrams):
        cls.bigrams_map = {word1:Counter() for word1, word2 in bigrams}
        for bigram in bigrams:
            cls.bigrams_map[bigram[0]].update([bigram[1]])
        return cls.bigrams_map

    @classmethod
    def get_suggestions(cls, s1, s2):
        suggestions = {word:freq for word, freq in cls.bigrams_map[s1].items() if word.startswith(s2)}
        return Counter(suggestions).most_common(10)
        
    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                inp = inp.lower()
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    @classmethod
    def run_train(cls, work_dir):
        cls.unigrams, bigrams = cls.load_training_data()
        print('Halfway!')
        cls.bigrams_map = cls.generate_bigram_map(bigrams)
        print('Done Training')
        # models.train_models(''.join(data))

    @classmethod
    def sort_tuple(cls, tup): 
        tup.sort(key = lambda x: x[1], reverse = True)  
        return tup  

    @classmethod
    def generate_possibilities(cls, current_string):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        suggestions = []
        for letter in alphabet:
            generate_output = cls.generate_all(current_string, letter) 
            for more_output in generate_output:
                if more_output not in suggestions:
                    suggestions.append(more_output)
        return sort_tuple(suggestions)[:3]

    @classmethod
    def generate_all(cls, current_string, next_char):
        top_tup_words = []
        seen_characters = set()

        if current_string in cls.bigrams_map: 
            words = cls.get_suggestions(current_string, next_char)
            for tup_word in words:
                if len(top_tup_words) == 3:
                    break
                if len(tup_word[0]) > len(next_char):
                    if tup_word[0][len(next_char)] not in seen_characters:
                        seen_characters.add(tup_word[0][len(next_char)])
                        top_tup_words.append(tup_word)
        else:
            top_tup_words = [('the', 'cool'), ('hi', 'super'), ('eat', 'food')]
        
        return top_tup_words # [(w, s). ....]

    @classmethod
    def generate_word(cls, s2):
        seen_characters = set()
        top_tup_words = []

        words = [(k, v) for k, v in cls.unigrams.most_common() if k.startswith(s2) and k != s2]
        for tup_word in words:
            cur_char = tup_word[0]
            if len(top_tup_words) == 3:
                break
            if len(cur_char) > len(s2):
                if cur_char[len(s2)] not in seen_characters:
                    seen_characters.add(cur_char[len(s2)])
                    top_tup_words.append(tup_word)
        return top_tup_words

    @classmethod
    def get_characters(cls, lst, s2):
        output_list = []
        for word in lst:
            curr_char = word[0]
            if len(curr_char) > len(s2):
                output_list.append(curr_char[len(s2)])
        if len(output_list) == 0:  # check if empty, no words
            return ['o', 'o', 'v']
        return output_list

    @classmethod    
    def predict(cls, s1, s2):
        if len(s1) == 0:
            suggestions = cls.generate_word(s2)
        elif len(s2) == 0:
            suggestions = cls.generate_possibilities(s1)
        else:
            suggestions = cls.generate_all(s1, s2)

        # Predict without any history
        if len(suggestions) < 3:
            new_suggestions = cls.generate_word(s2)
            if len(new_suggestions) == 0: 
                return ['o', 'o', 'v']

            for i in range(3):
                if len(new_suggestions) - 1 < i:
                    suggestions.append((s2 + "s", 0))
                elif new_suggestions[i][0] not in [j[0] for j in suggestions]:
                    suggestions.append(new_suggestions[i])
                if len(suggestions) == 3:
                    break

        return cls.get_characters(suggestions, s2)

    def run_pred(self, data):
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

    def save(self, work_dir):
        pickle.dump({'unigram_map': self.unigrams,
                    'bigrams_map': self.bigrams_map},
                    open(os.path.join(work_dir, 'model.checkpoint'), 'wb'),
                    protocol=2)

        # models.save_models(os.path.join(work_dir, 'model.checkpoint'))

    @classmethod
    def load(cls, work_dir):
        models = pickle.load(open(os.path.join(work_dir, 'model.checkpoint'),'rb'))

        return MyModel(models['unigram_map'], models['bigrams_map'])


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)

        print('Instantiating model')
        model = MyModel()
        print('Training')
        model.run_train(args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
