import gzip
from collections import *
import math

tokenss = defaultdict(Counter)
total_tokss = Counter()

for lang in "afr-1953 aln-aln arb-arb arz-arz ayr-1997 ayr-2011 bba-bba ben-common ben-mussolmani bqc-bqc bul-bul bul-veren cac-ixtatan cak-central2003 ceb-bugna2009 ceb-bugna ceb-pinadayag ces-ekumenicky ces-kralicka cmn-sf_ncv-zefania cnh-cnh cym-morgan1804 dan-1931 deu-elberfelder1871 deu-elberfelder1905 deu-freebible deu-gruenewalder deu-luther1545letztehand deu-luther1912 deu-neue deu-pattloch deu-schlachter deu-textbibel deu-zuercher ell-modern2009 eng-darby eng-kingjames eng-literal eng-newsimplified epo-epo fin-1766 fin-1933 fin-1992 fra-bonnet fra-crampon fra-darby fra-davidmartin fra-jerusalem2004 fra-kingjames fra-louissegond fra-ostervald1867 fra-paroledevie fra-perret fra-pirotclamer guj-guj gur-frafra hat-1985 hat-1999 hrv-hrv hun-2005 hun-karoli ind-suciinjil ind-terjemahanbaru ita-2009 ita-diodati ita-nuovadiodati1991 ita-riveduta kek-1988 kek-2005 kjb-kjb lat-novavulgata lit-lit mah-mah mam-northern mri-mri mya-mya nld-nld nor-nor nor-student plt-romancatholic poh-eastern por-almeidaatualizada por-almeidarevista por-paratodos qub-qub quh-1993 quy-quy quz-quz ron-cornilescu rus-synodal som-som tbz-tbz tcw-tcw tgl-1905 tlh-klingon tpi-tpi tpm-tpm ukr-1962 ukr-2009 vie-1926compounds vie-1926nocompounds vie-2002 wal-wal wbm-wbm xho-xho zom-zom".split():
# for lang in "bg cs da de el en es et fi fr hu it lt lv nl pl pt ro sk sl sv".split():
  with gzip.open(f"europarl.21/splits/{lang}.train.charunked.gz", 'rt', encoding = 'utf-8') as f:
    for line in f:
      toks = list(line.split())
      tokenss[lang[:3]].update(toks)
      total_tokss[lang[:3]] += len(toks)

for lang in sorted(list(tokenss.keys())):
  entropy = 0
  for tok, freq in tokenss[lang].most_common():
    p = freq / total_tokss[lang]
    entropy -= p * math.log2(p)
  print(lang, entropy)
