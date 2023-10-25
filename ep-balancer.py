import random
import os

with open("europarl_native.10/native_lang") as f:
  native_langs = f.read().splitlines()

line2enwords = []
with open("europarl_native.10/en") as en_file:
  for line in en_file:
    line2enwords.append(len(list(line.split())))

lang2natives = {lang: [idx for idx, nl in enumerate(native_langs) if nl == lang] for lang in set(native_langs)}

for balanced_language in ["da", "de", "en", "es", "fi", "fr", "it", "nl", "pt", "sv"]:
  # Choose which line indices to use
  native_indices = [idx for idx, nl in enumerate(native_langs) if nl == balanced_language]
  transl_indices = [idx for idx, nl in enumerate(native_langs) if nl != balanced_language]
  random.shuffle(native_indices)
  random.shuffle(transl_indices)
  chosen_idxs = []
  total_en_words_native = 0
  for idx in native_indices:
    chosen_idxs.append(idx)
    total_en_words_native += line2enwords[idx]
    if total_en_words_native >= 500000:
      break
  total_en_words_transl = 0
  for idx in transl_indices:
    chosen_idxs.append(idx)
    total_en_words_transl += line2enwords[idx]
    if total_en_words_transl >= total_en_words_native:
      break
  random.shuffle(chosen_idxs)
  chosen_idxs_set = set(chosen_idxs)
  # Generate the corresponding files!
  os.mkdir("europarl_balanced_" + balanced_language)
  for output_language in ["da", "de", "en", "es", "fi", "fr", "it", "nl", "pt", "sv", "native_lang"]:
    # Read in sequentially
    chosen_lines = {}
    with open("europarl_native.10/" + output_language) as en_file:
      for line_nr, line in enumerate(en_file):
        if line_nr in chosen_idxs_set:
          chosen_lines[line_nr] = line
    # Print in new order
    with open("europarl_balanced_" + balanced_language + "/" + output_language, 'w') as outfile:
      for line_nr in chosen_idxs:
        outfile.write(chosen_lines[line_nr])
