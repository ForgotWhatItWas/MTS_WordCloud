# -*- coding: utf-8 -*-

import gensim.downloader
import requests, re, nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pymorphy3
import spacy
from nltk.corpus import stopwords
import jamspell
from navec import Navec
import fasttext as ft
from scipy import spatial
import os
from flask import Flask, request, abort, send_file

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
STOPWORDS = set(spacy.lang.en.stop_words.STOP_WORDS)
jsp_eng = jamspell.TSpellCorrector()
assert jsp_eng.LoadLangModel('en.bin')
jsp = jamspell.TSpellCorrector()
assert jsp.LoadLangModel('ru_small.bin')
morphy = pymorphy3.MorphAnalyzer()
navec = Navec.load('/content/navec_hudlit_v1_12B_500K_300d_100q.tar')
model = ft.load_model('ft_native_300_ru_twitter_nltk_word_tokenize.bin')
glove_vectors = gensim.downloader.load('word2vec-google-news-300')
UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'txt'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_synonyms(word_freq):

  # Step 1: Initialize an empty Counter for the final unique words and their frequencies
  final_counter = Counter()

  # Convert the input Counter to a list of (word, frequency) tuples
  words_and_freqs = list(word_freq.items())
  # Step 2: Start processing the words
  while words_and_freqs:
      # Take the first word and frequency from the list
      current_word, current_freq = words_and_freqs.pop(0)
      if re.search(r"[а-яёА-ЯЁ]",current_word):
        s11 = model.get_word_vector(current_word)
        s12=navec.get(current_word, navec['<unk>'])
      # Step 4: Compare the current word with the remaining words
        for word, freq in words_and_freqs[:]:
            s01 = model.get_word_vector(word)
            s02=navec.get(word, navec['<unk>'])
            if (1 - spatial.distance.cosine(s01, s11))>0.7 or (1 - spatial.distance.cosine(s02, s12))>0.421:
                # If the word is the same, add its frequency to the current one
                current_freq += freq
                # Remove the word from the list (it's already processed)
                words_and_freqs.remove((word, freq))
        # Step 5: Update the frequency of the current word in the final_counter
        final_counter[current_word] = current_freq
      else:
        for word, freq in words_and_freqs[:]:
          try:
            if glove_vectors.similarity(current_word, word)>0.4 and (current_word !="good" and word !="bad") :
             # If the word is the same, add its frequency to the current one
                current_freq += freq
                # Remove the word from the list (it's already processed)
                words_and_freqs.remove((word, freq))
          except KeyError:
            continue
        # Step 5: Update the frequency of the current word in the final_counter
        final_counter[current_word] = current_freq

  return final_counter

# Normalize and lemmatize text
def clean_and_lemmatize_eng(text):
  with open("badwords.txt", 'r') as badwords:
    bad_w = badwords.read().splitlines()
    doc = nlp(jsp_eng.FixFragment(text))  # lemmatise+normalise
    lemmas = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in STOPWORDS and token.lemma_ not in bad_w]
  return lemmas

  # Further clean, fix grammar and lemmatize text
def clean_and_lemmatize(text):
  if re.search(r"[а-яёА-ЯЁ]",text):
    with open("words.txt", 'r') as bad:
      words = word_tokenize(jsp.FixFragment(text))
      no_blank=[]
      for word in words:
          if word!='' and word!=' ':
              no_blank.append(word)

      #Норма
      for k, word in enumerate(words):
        words[k]=morphy.parse(word)[0].normalized.word

      #Удаление стоп-слов
      words_without_sw=[]
      bad_lines = bad.read().splitlines()
      for word in words:
          if word not in bad_lines and word not in stopwords.words("russian"):
            words_without_sw.append(word)
      return words_without_sw

  else:
    clean_line = clean_and_lemmatize_eng(text)
    return clean_line

# Main processing function
def process_txt_file(input_path, output_path):

    counter = Counter()
    # Precompiled regex pattern to remove all punctuation (except dash), English letters, and digits, with Unicode support
    pattern = re.compile(r'[0-9!"#$%&\'()*+,./:;<=>?@[\\\]^_`{|}~“”‘’„…-]')

    # Open and process the file
    with open(input_path, 'r') as file:
        lines = [pattern.sub(' ', line).lower().strip() for line in file.read().splitlines()]

    # Iterate over the lines (skipping the first question line)
    for line in lines[:]:
        clean_line = clean_and_lemmatize(line)  # Clean and lemmatize the line
        if clean_line:
          for keyword in clean_line:
            counter[keyword] += 1

    filtered_word_freq = get_synonyms(counter)

    # Write results to output file
    with open(output_path, 'w') as file:
        for word, count in filtered_word_freq.items():
            file.write(f'{word}: {count}\n')

# Function to check if file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def wrds():
    # Check if a file part is in the request
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    # Check if the file is selected
    if file.filename == '':
        return 'No selected file', 40

    # Check file type and save it as 'text.txt'
    if file and allowed_file(file.filename):
        filename = 'text.txt'  # Always save as 'text.txt'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            return f'Error saving file: {str(e)}', 500

        # Process the file and generate 'output.txt'
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'output.txt')
        try:
            process_txt_file(filepath, output_filepath)
        except Exception as e:
            return f'Error processing file: {str(e)}', 500

        # Return the processed output.txt file for download
        try:
            return send_file(output_filepath, as_attachment=True, download_name='output.txt')
        except FileNotFoundError:
            abort(404, description="Output file not found")
    else:
        return 'File type not allowed', 400

# Main entry point
if __name__ == '__main__':
    app.run()
