import os
import string
import random
import re
import numpy
from numpy import array
from pickle import dump

# !pip install autocomplete
import autocomplete
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

autocomplete.load()
from autocomplete import models

def build_model():
  all_contents = []
  for file in os.listdir('data/train'):
      with open('data/train/' + file, 'r') as f:
          content = f.read()
          content = content.replace("<br /><br />", "")
          content = re.sub('[^A-Za-z\']', " ", content)
          content = re.sub("\s\s+" , " ", content)
          content = content.strip()
          content = content.lower()
          all_contents.append(content)
  print(''.join(all_contents))
  models.train_models(''.join(all_contents))
  models.save_models()

def read_files():
  tokenized_files = []
  for file in os.listdir('data/train'):
      with open('data/train/' + file, 'r') as f:
          content = f.read()
          content = content.replace("<br /><br />", "")
          content = re.sub('[^A-Za-z\']', " ", content)
          content = content.strip()
          content = content.lower()
      tokens = content.split()
      tokenized_files.append(tokens)
  return tokenized_files

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
        print(more_output, suggestions)
        suggestions.append(more_output)
  return sort_tuple(suggestions)[:3]

def generate_all(current_string, next_char):
  top_tup_words = []
  seen_characters = set()
  words = autocomplete.predict(current_string, next_char)
  for tup_word in words:
    if len(top_tup_words) == 3:
      break
    elif tup_word[0][0] not in seen_characters:
      seen_characters.add(tup_word[0][0])
      top_tup_words.append(tup_word)
  return top_tup_words # [(w, s). ....]

def get_characters(list):
  output_list = []
  for word in list:
    output_list.append(word[0][0])
  return output_list
    
def predict(s1, s2):
  if len(s2) == 0:
    suggestions = generate_possibilities(s1)
  else:
    suggestions = generate_all(s1, s2)
  return get_characters(suggestions)

# if __name__ == "__main__"
build_model()
predict('comic', '')