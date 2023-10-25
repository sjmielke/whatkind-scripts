import os
import numpy as np
import scipy
import time
import gzip
import itertools
import gurobipy

# Construct presence matrix

bible_files = sorted(os.listdir(BIBLE_PATH))
bible_langs = np.array([bib[:3] for bib in bible_files])

presence = np.zeros((len(bible_files), 43904), dtype = int) + 2
for bibleno, filename in enumerate(bible_files):
print("Reading", filename)
with gzip.open(BIBLE_PATH + '/' + filename, 'rt') as file:
    empty = True
    for lineno, line in enumerate(file):
    empty = False
    presence[bibleno, lineno] = 0 if line.strip() in ["", "BLANK"] else 1
    if empty:
    presence[bibleno, :] = 0
np.save('ALL_PRESENCES', presence)

# Cull too small bibles

bible_lengths = presence.sum(axis=1)
acceptable_bibles = (bible_lengths >= 20000)
print("Selecting from", sum(acceptable_bibles), "bibles")
bible_lengths = bible_lengths[acceptable_bibles]
presence = presence[acceptable_bibles, :]
bible_langs = bible_langs[acceptable_bibles]

# Now set up the Gurobi ILP

bible_langs_unique = sorted(list(set(bible_langs)))

m = gurobipy.Model("bibleselection")
g_bible = [m.addVar(vtype = gurobipy.GRB.BINARY, name = "bible_" + str(i)) for i in range(presence.shape[0])]
g_language = [m.addVar(vtype = gurobipy.GRB.BINARY, name = "lang_" + l) for l in bible_langs_unique]
g_verse = [m.addVar(vtype = gurobipy.GRB.BINARY, name = "verse_" + str(i)) for i in range(presence.shape[1])]
m.update()

m.setObjective(0.000001 * gurobipy.quicksum(g_bible) + gurobipy.quicksum(g_language) * gurobipy.quicksum(g_verse), gurobipy.GRB.MAXIMIZE)

for var, lang in zip(g_language, bible_langs_unique):
  bibles_in_this_language = [v for v, l in zip(g_bible, bible_langs) if l == lang]
  m.addConstr(var <= gurobipy.quicksum(bibles_in_this_language), "existsbibforlang_" + lang)

for i_bible in range(presence.shape[0]):
  for i_verse in range(presence.shape[1]):
    if not presence[i_bible, i_verse]:
      m.addConstr(g_bible[i_bible] + g_verse[i_verse] <= 1, "disallow_" + str(i_bible) + "_" + str(i_verse))

m.optimize()
m.printAttr("X", "bible*")

# This will print out the solution! For me it happened to look like on of these vectors:

bcs = [
  np.array([0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,24,25,26,27,28,29,31,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,62,63,64,65,73,74,75,77,78,79,80,81,82,84,85,86,88,90,91,92,94,95,96,97,98,100,101,102,103,104,105,106,107,108,109,110,111,113,114,115,116,117,118,120,121,122,123,124,125,126,128,130]),  # bibles x verses
  np.array([0,1,2,3,5,7,9,10,11,12,13,14,16,18,20,21,22,24,25,26,27,32,33,39,40,42,44,45,50,54,57,58,62,63,65,73,74,77,78,80,81,82,84,86,88,90,91,92,94,95,96,98,100,101,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,121,123,125,126,128,130]) # langs x verses
]

# We can continue processing with these:

for bc in bcs:
  print(list(bible_langs[bc]))
  parallelity = presence[bc, :].sum(axis = 0)
  nbibs = len(bc)
  nlangs = len(set(bible_langs[bc]))
  nverses = (parallelity == nbibs).sum()
  print(nbibs, nlangs, nverses, nbibs * nverses, nlangs * nverses)

# Now extract the "best" set: I'm choosing "bibles x verses" cause it only has one less language: Swedish.

chosen_bible_files = np.array(bible_files)[acceptable_bibles][bcs[0]]
chosen_presence = presence[bcs[0], :]
chosen_parallelity = presence[bcs[0], :].sum(axis = 0)
chosen_includability = chosen_parallelity == len(bcs[0])
for filename in chosen_bible_files:
  assert filename.endswith("-v1.txt.gz")
  assert 'bpe' not in filename
  assert 'char' not in filename
  biblename = filename[:-len("-v1.txt.gz")].replace("-x-bible", "")
  print("Reading", biblename, "(", filename, ")")
  with gzip.open(BIBLE_PATH + '/' + filename, 'rt') as infile:
    with gzip.open("./datasets/chosen_bibles/" + biblename + ".txt.gz", 'wt') as outfile:
      for lineno, line in enumerate(infile):
        if chosen_includability[lineno]:
          print(line.strip(), file = outfile)
