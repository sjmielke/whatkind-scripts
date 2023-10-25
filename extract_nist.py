import numpy as np
import xml.etree.ElementTree

import reversible_tokenize

path = "/home/sjm/projects/JHU/variance-and-variation/NIST OpenMT Eval/"

total_rails = ["", "", "", ""]

for filename in """
LDC2010T10/nist_openmt_2002/data/mt02_arabic_evalset_v0-ref.xml
LDC2010T10/nist_openmt_2002/data/mt02_chinese_evalset_v0-ref.xml
LDC2010T11/nist_openmt_2003/data/mt03_arabic_evalset_v0-ref.xml
LDC2010T11/nist_openmt_2003/data/mt03_chinese_evalset_v0-ref.xml
LDC2010T12/nist_openmt_2004/data/mt04_arabic_evalset_v0-ref.xml
LDC2010T12/nist_openmt_2004/data/mt04_chinese_evalset_v1-ref.xml
LDC2010T14/nist_openmt_2005/data/mt05_arabic_evalset_v0-ref.xml
LDC2010T14/nist_openmt_2005/data/mt05_chinese_evalset_v0-ref.xml
LDC2010T17/nist_openmt_2006/data/mt06_arabic_evalset_nist_part_v1-ref.xml
LDC2010T17/nist_openmt_2006/data/mt06_chinese_evalset_nist_part_v1-ref.xml
LDC2010T21/nist_openmt_2008/data/mt08_arabic_evalset_current_v0-ref.xml
LDC2010T21/nist_openmt_2008/data/mt08_chinese_evalset_current_v0-ref.xml
LDC2010T21/nist_openmt_2008/data/mt08_urdu_evalset_current_v0-ref.xml
LDC2010T23/nist_openmt_2009/data/mt09_arabic_evalset_current_v3-ref.xml
LDC2010T23/nist_openmt_2009/data/mt09_urdu_evalset_current_v3-ref.xml
LDC2013T03/nist_openmt_2012_curr_test/data/OpenMT12_Current_chi2eng-ref.xml
LDC2013T03/nist_openmt_2012_curr_test/data/OpenMT12_Current_chi2eng_RestrictedDomain-ref.xml
LDC2013T07/nist_openmt_2008-2012_prog_tests/data/OpenMT08-12_Progress_ara2eng-ref.xml
LDC2013T07/nist_openmt_2008-2012_prog_tests/data/OpenMT08-12_Progress_chi2eng-ref.xml
""".strip().splitlines():

  root = xml.etree.ElementTree.parse(path + filename).getroot()

  refsets = []
  for refset in root:
    print(refset.attrib['refid'])
    docs = []
    for doc in refset:
      lines = []
      for seg in doc:
        if seg.tag == "hl":
          seg = seg[0]
        assert seg.tag == "seg"
        assert list(seg) == []  # assert no child tags, already know that there are no entities, these are taken care of by ElementTree
        assert '\n' not in seg.text  # assert no line breaks, will insert these later!
        lines.append(seg.text.strip())
      docs.append(lines)
    refsets.append(docs)

  assert all([len(refsets[i]) == len(refsets[i + 1]) for i in range(len(refsets) - 1)])  # assert all refsets have the same (number of) documents
  lenss = [[len(doc) for doc in docs] for docs in refsets]
  assert all([[len(doc) for doc in refsets[i]] == [len(doc) for doc in refsets[i + 1]] for i in range(len(refsets) - 1)])  # assert all docs have the same length across refsets

  partexts = ["\n".join(["\n".join(doc) for doc in refset]) for refset in refsets]
  assert len(partexts) == 4  # weird but thats the data, so its fine
  print([t[:40] for t in partexts])

  for i in range(4):
    total_rails[i] += partexts[i]

for i, t in enumerate(total_rails):
  with open(path + "total_rails_" + str(i), 'w') as file:
    s = total_rails[i]
    reversible_tokenize.check_for_at(s)  # prints errors if ambiguous
    s = reversible_tokenize.tokenize(s)
    print(s, file = file)
