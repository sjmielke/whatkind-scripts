import collections
import difflib
import os
import re
import shutil
import numpy as np
import xml.etree.ElementTree

import reversible_tokenize

ALL_LANGS = set("en fr it nl pt es da de sv fi el cs et lt sk lv sl pl hu ro bg".split())

FOUND_NATIVE_LANGS = collections.Counter()
NOT_FOUND_NATIVE_LANGS = collections.Counter()
OUTPUT_LANGS = collections.Counter()
MULTINESS = collections.Counter()
BROKEN = 0

ALL_PARALLEL_PS = collections.defaultdict(list)
BEFORE = []
AFTER = []

for xmlfile in sorted(os.listdir('../costep/')):
# for xmlfile in ['1999-09-15.xml']:
  print(xmlfile, end="\r")
  root = xml.etree.ElementTree.parse('../costep/' + xmlfile).getroot()  # blows size up by factor of 3 -- feasible.
  for chapter_id, chapter in enumerate(root):
    # print("Chapter", chapter_id + 1)
    # Local variables to be able to skip "faulty" chapters (i.e., chapters with seemingly misaligned turns)
    ok_chapter = True
    found_native_langs = collections.Counter()
    not_found_native_langs = collections.Counter()
    output_langs = collections.Counter()
    multiness = collections.Counter()
    broken = 0
    # Gather the children, obeying order.
    headline_ok = True
    headlines = []
    speakers = []
    for turn_or_headline in chapter:
      if turn_or_headline.tag == 'headline' and headline_ok:
        headlines.append(turn_or_headline)
      elif turn_or_headline.tag == 'turn':
        [speaker] = list(turn_or_headline) # every turn only consists of a single speaker
        assert speaker.tag == 'speaker'
        speakers.append(speaker)
        headline_ok = False
      else:
        raise Exception("Illegal chapter content!")
    # Look at data
    # print(len(headlines), "headlines")
    # print("Turns:", [len(s) for s in speakers])
    for speaker_id, speaker in enumerate(speakers):
      # print("Turn/speaker", speaker_id + 1)
      # Some assertions
      for text in speaker:
        assert text.tag == 'text', "Illegal speaker content: " + str(text)
        for p in text:
          assert p.tag == 'p'
          assert p.attrib['type'] in ['comment', 'speech']
        # Forget it. They fucked it up, nothing we can do.
        # assert (len(text) > 0 and 'empty' not in text.attrib) or (len(text) == 0 and 'empty' in text.attrib and text.attrib['empty'] == 'yes'), str(text.attrib) + " -- len:" + str(len(text)) + xmlfile
      # Extract useful data
      lang_texts = [
        (text.attrib["language"], [p for p in text if p.attrib['type'] == 'speech'])
        for text in speaker
        if 'empty' not in text.attrib or len(text) == 0
      ]
      # Get the speaker's native language
      if speaker.attrib['president'] == 'yes':
        speaker_native_lang = "president"
      elif "language" in speaker.attrib:
        speaker_native_lang = speaker.attrib["language"]
      else:
        speaker_native_lang = "unknown"
      # Now count stuff!
      if lang_texts != []:
        median_pars = int(np.median([len(ps) for lang, ps in lang_texts]))
        # Count individual languages
        got_native = False
        for lang, ps in lang_texts:
          if lang == speaker_native_lang:
            found_native_langs[lang] += len(ps)
            got_native = True
          output_langs[lang] += len(ps)
        if not got_native:
          # if speaker_native_lang == 'en':
          #   raise Exception("Didn't find", speaker_native_lang, "in", [l for l, _ in lang_texts], "in", xmlfile, speaker.attrib["extra"])
          not_found_native_langs[speaker_native_lang] += median_pars
        # Count parallelity
        try:
          pars = int(np.unique([len(ps) for lang, ps in lang_texts]))
          multiness[len(lang_texts)] += pars
          # Write it out!
          for lang, ps in lang_texts:
            ALL_PARALLEL_PS[lang] += [(speaker_native_lang, p) for p in ps]
          for missing_lang in ALL_LANGS - set((lang for lang, ps in lang_texts)):
            ALL_PARALLEL_PS[missing_lang] += [("", None)] * pars
        except TypeError:
          # print("Unequal number of paragraphs:", nums_speech_paragraphs)
          broken += median_pars
          ok_chapter = False

    # Now update global counters
    OUTPUT_LANGS.update(output_langs)
    BROKEN += broken
    if ok_chapter:
      FOUND_NATIVE_LANGS.update(found_native_langs)
      NOT_FOUND_NATIVE_LANGS.update(not_found_native_langs)
      MULTINESS.update(multiness)
    else:
      BROKEN += sum(multiness.values())


print("found native langs", FOUND_NATIVE_LANGS)
print("not found native langs", NOT_FOUND_NATIVE_LANGS)
print("output langs", OUTPUT_LANGS)
print("nlangs/multiness", MULTINESS)
print("broken median", BROKEN)

print("Total extracted paragraphs:", {k: len(v) for k, v in ALL_PARALLEL_PS.items()})

tags = []
for lang, ps in ALL_PARALLEL_PS.items():
  for l, p in ps:
    if p is not None:
      tags += [el.tag for el in p.iter()]
print("Inner tags:", set(tags))

def de_xml_ify(root):
  el_string = root.text if root.text else ""  # first the part until the first child tag
  for el in list(root):
    if el.tag == "ellipsis":
      assert not el and not el.text  # should be childless and empty
      el_string += "…"
    elif el.tag in ["url", "report", "procedure", "ref", "n"]:
      el_string += de_xml_ify(el)
    elif el.tag == "quote":
      if 'start' not in el.attrib or 'end' not in el.attrib:
        print("malformed quote: ", xml.etree.ElementTree.tostring(root, encoding = 'utf-8').decode('utf-8'))
      el_string += (el.attrib['start'] if 'start' in el.attrib else '"')
      el_string += de_xml_ify(el)
      el_string += (el.attrib['end'] if 'end' in el.attrib else '"')
    else:
      raise Exception("Unknown tag encountered while de_xml_fy-ing " + str(root) + ": " + str(el) + ", fully expanded:", xml.etree.ElementTree.tostring(root, encoding = 'utf-8').decode('utf-8'))
    # append the subsequent string
    el_string += el.tail if el.tail else ""
  return el_string



# for langset in [set("en fr it nl pt es da de sv fi".split()), ALL_LANGS]:
# for langset in [set("en fr it nl pt es da de sv fi".split())]:
for langset in [ALL_LANGS]:
  # Set up directories
  if os.path.isdir("europarl_native." + str(len(langset))):
    shutil.rmtree("europarl_native." + str(len(langset)))
  os.mkdir("europarl_native." + str(len(langset)))
  # Now we filter the parallel sentences to those that exist in all languages
  interesting_languages = [ps for lang, ps in ALL_PARALLEL_PS.items() if lang in langset]
  ok_indices = set([i for i, p_tuple in enumerate(zip(*interesting_languages)) if all([p is not None for l, p in p_tuple])])
  for i_lang, lang in enumerate(sorted(list(langset))):
    ps = ALL_PARALLEL_PS[lang]
    with open("europarl_native." + str(len(langset)) + "/" + lang, 'a') as file, open("europarl_native." + str(len(langset)) + "/native_lang", 'a') as native_file:
      # print()
      for i_p, (native_lang, p) in enumerate(ps):
        if i_p not in ok_indices:
          continue
        if i_p % 500 == 0:
          print("Writing out language", lang, ",", i_lang+1, "/", len(langset), "paragraph", i_p+1, "/", len(ps), end="\r")
        # Clean the XML stuff
        p_string = de_xml_ify(p).strip()
        # Replace newlines to ensure alignedness in line format
        p_string = p_string.replace('\n', ' ')
        # # Sanity check, are changes good?
        # cheap_conversion = xml.etree.ElementTree.tostring(p, encoding = 'utf-8').decode('utf-8').replace('<p type="speech">', '', 1).replace('</p>', '', 1).replace('&amp;', '&').strip()
        # if p_string != cheap_conversion:
        #   BEFORE.append(cheap_conversion)
        #   AFTER.append(p_string)
        # Tokenize
        reversible_tokenize.check_for_at(p_string)  # prints errors if ambiguous
        p_string = reversible_tokenize.tokenize(p_string)
        assert '\n' not in p_string  # should have fixed this above
        print(p_string, file = file)
        if i_lang == 0:
          print(native_lang, file = native_file)
  # print("Now generating a big-ass diff table.")
  # with open('/tmp/diff.htm', 'w') as f:
  #   print(difflib.HtmlDiff(charjunk = None).make_file('\n'.join(BEFORE).split(), '\n'.join(AFTER).split(), context = True).replace('td.diff_header {text-align:right}', 'td.diff_header {text-align:right}\ntd {max-width:300px !important;}'), file = f)
