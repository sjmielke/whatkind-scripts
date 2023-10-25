#!/usr/bin/env bash
set -e

# Assumes we are in a directory that contains:
# - tok/: tokenized files called de, en, etc.
# Will create new directories:
# - splits/: train/dev/test splits, charunked versions
# - bpe/: the BPE codes and BPEified splits
# - kenlm/: KenLM model files and results on test data
# - rnnlm/: RNNLM running scripts, models, logs, etc.

MAINDIR="$(pwd)"

# LANGS="bg cs da de el en es et fi fr hu it lt lv nl pl pt ro sk sl sv"
# LANGS="afr-1953 aln-aln arb-arb arz-arz ayr-1997 ayr-2011 bba-bba ben-common ben-mussolmani bqc-bqc bul-bul bul-veren cac-ixtatan cak-central2003 ceb-bugna2009 ceb-bugna ceb-pinadayag ces-ekumenicky ces-kralicka cmn-sf_ncv-zefania cnh-cnh cym-morgan1804 dan-1931 deu-elberfelder1871 deu-elberfelder1905 deu-freebible deu-gruenewalder deu-luther1545letztehand deu-luther1912 deu-neue deu-pattloch deu-schlachter deu-textbibel deu-zuercher ell-modern2009 eng-darby eng-kingjames eng-literal eng-newsimplified epo-epo fin-1766 fin-1933 fin-1992 fra-bonnet fra-crampon fra-darby fra-davidmartin fra-jerusalem2004 fra-kingjames fra-louissegond fra-ostervald1867 fra-paroledevie fra-perret fra-pirotclamer guj-guj gur-frafra hat-1985 hat-1999 hrv-hrv hun-2005 hun-karoli ind-suciinjil ind-terjemahanbaru ita-2009 ita-diodati ita-nuovadiodati1991 ita-riveduta kek-1988 kek-2005 kjb-kjb lat-novavulgata lit-lit mah-mah mam-northern mri-mri mya-mya nld-nld nor-nor nor-student plt-romancatholic poh-eastern por-almeidaatualizada por-almeidarevista por-paratodos qub-qub quh-1993 quy-quy quz-quz ron-cornilescu rus-synodal som-som tbz-tbz tcw-tcw tgl-1905 tlh-klingon tpi-tpi tpm-tpm ukr-1962 ukr-2009 vie-1926compounds vie-1926nocompounds vie-2002 wal-wal wbm-wbm xho-xho zom-zom"
# LANGS="nist_rail_1 nist_rail_2 nist_rail_3 nist_rail_4"
LANGS="da de en es fi fr it nl pt sv"
LANGS="bg-small cs-small da-small de-small el-small en-small es-small et-small fi-small fr-small hu-small it-small lt-small lv-small nl-small pl-small pt-small ro-small sk-small sl-small sv-small"

# # Making small languages
# for lang in bg cs da de el en es et fi fr hu it lt lv nl pl pt ro sk sl sv; do
#     zcat "splits/${lang}.train.gz" | awk "int(NR / 20) % 10 == 0" | gzip > "splits/${lang}-small.train.gz"
#     cp splits/${lang}{,-small}.dev.gz
#     cp splits/${lang}{,-small}.test.gz
# done

# LANGS_AND_REV="${LANGS} ${LANGS// /-rev }-rev"
LANGS_AND_REV="${LANGS}"

# BPE_FACTORS="$(seq 0 0.1 1)"
BPE_FACTORS="0.4"

case "$1" in
  lanayru)
    BPE_PATH="/home/sjm/projects/ISI/subword-nmt/sennrich_bpe/"
    MOSES_PATH="/home/sjm/programming/mosesdecoder/"
    RNNLM_CALL="unbuffer python /home/sjm/projects/JHU/inlm/awd-lstm-lm/main.py --device cpu"
    RNNLM_RUNNER="bash"
    ;;

  clspgrid)
    BPE_PATH="/home/smielke/subword-nmt/"
    MOSES_PATH="/home/smielke/mosesdecoder/"
    RNNLM_CALL='CUDA_VISIBLE_DEVICES=`free-gpu` unbuffer python /home/smielke/inlm/awd-lstm-lm/main.py --device cuda'
    RNNLM_RUNNER='qsub -l hostname=c*,gpu=1 -q g.q -l mem_free=10G,ram_free=10G'
    ;;

  marcc)
    BPE_PATH="/scratch/users/smielke2@jhu.edu/subword-nmt"
    MOSES_PATH=""
    RNNLM_CALL="ml singularity; TCLLIBPATH=/home-4/smielke2@jhu.edu/local/lib LC_ALL='C.UTF-8' unbuffer singularity exec --nv /scratch/users/smielke2\@jhu.edu/pytorch.0.4.1.simg python /scratch/users/smielke2\@jhu.edu/inlm/awd-lstm-lm/main.py --device cuda"
    RNNLM_RUNNER="sbatch -p gpuk80 --gres=gpu:1 --mem=50G --time=48:00:00"
    ;;

  *)
    echo $"Usage: $0 {lanayru|clspgrid|marcc} [kenlm|rnnlmbpe|rnnlmchar]"
    exit 1
esac

# Make train/test splits
mkdir -p "splits"
for lang in ${LANGS}; do
  if ! [ -s splits/${lang}.train.gz -a -s splits/${lang}.dev.gz -a -s splits/${lang}.test.gz ]; then
    echo "Splitting for ${lang}"
    # Dev set: 5 lines, every 30 lines.
    IS_DEV="NR % 30 == 1 || NR % 30 == 2 || NR % 30 == 3 || NR % 30 == 4 || NR % 30 == 5"
    # Test/measuring set: 5 lines, every 30 lines.
    IS_TEST="NR % 30 == 6 || NR % 30 == 7 || NR % 30 == 8 || NR % 30 == 9 || NR % 30 == 10"
    if [ ${#lang} -eq 2 ]; then
      INFILE="tok/${lang}.gz"
    else
      INFILE="tok/${lang}.txt.gz"
    fi
    zcat ${INFILE} | awk "! (( $IS_DEV ) || ( $IS_TEST ))" | gzip > splits/${lang}.train.gz
    zcat ${INFILE} | awk "$IS_DEV" | gzip > splits/${lang}.dev.gz
    zcat ${INFILE} | awk "$IS_TEST" | gzip > splits/${lang}.test.gz
  fi
done

# Char-UNK with ◊ (forward AND reversed)
for lang in ${LANGS}; do
  if ! [ -s splits/${lang}.train.charunked.gz -a -s splits/${lang}.dev.charunked.gz -a -s splits/${lang}.test.charunked.gz -a -s splits/${lang}-rev.train.charunked.gz -a -s splits/${lang}-rev.dev.charunked.gz -a -s splits/${lang}-rev.test.charunked.gz ]; then
    echo "Char-UNKing for ${lang}"
    python<<EOF
import collections
import gzip
with gzip.open("splits/${lang}.train.gz", 'rt', encoding = 'utf-8') as f:
  CHARSET = set([char for (char, count) in collections.Counter(f.read()).most_common() if count >= 25])
for setname in [".train", ".dev", ".test"]:
  with gzip.open("splits/${lang}" + setname + ".gz", 'rt', encoding = 'utf-8') as f, \
       gzip.open("splits/${lang}" + setname + ".charunked.gz", 'wt', encoding = 'utf-8') as fw, \
       gzip.open("splits/${lang}-rev" + setname + ".charunked.gz", 'wt', encoding = 'utf-8') as fw_rev:
    for line in f:
      fw.write(''.join(c if c in CHARSET or c.isspace() else '◊' for c in line))
      fw_rev.write(''.join(reversed(list(c if c in CHARSET or c.isspace() else '◊' for c in line))))
EOF
  fi
done

# Charify for char-level modeling
for lang in ${LANGS_AND_REV}; do
  if ! [ -s splits/${lang}.train.charunked.chars.gz -a -s splits/${lang}.dev.charunked.chars.gz -a -s splits/${lang}.test.charunked.chars.gz ]; then
    echo "Charifying for ${lang}"
    python<<EOF
import gzip
SPACE = '⁀'  # This specific character is used in main.evaluate (L505) for the loss summing of characters into words
for setname in [".train", ".dev", ".test"]:
  with gzip.open("splits/${lang}" + setname + ".charunked.gz", 'rt', encoding = 'utf-8') as f, \
       gzip.open("splits/${lang}" + setname + ".charunked.chars.gz", 'wt', encoding = 'utf-8') as fw:
    for line in f:
      print(' '.join(SPACE.join(line.split())), file = fw)
EOF
  fi
done

# BPE-ize
mkdir -p bpe
for lang in ${LANGS_AND_REV}; do
  for factor in ${BPE_FACTORS}; do
    if ! [ -s bpe/${lang}.train.charunked.bpe_${factor/./}.gz -a -s bpe/${lang}.dev.charunked.bpe_${factor/./}.gz -a -s bpe/${lang}.test.charunked.bpe_${factor/./}.gz ]; then
      echo "BPE ${factor} for ${lang}"
      # How many merges? Half the raw vocab size.
      num_types=$(zcat splits/${lang}.train.charunked.gz | tr ' ' '\n' | LC_ALL=C sort -u | wc -l)
      num_merges=$(echo "scale=0; ${num_types} * ${factor} / 1" | bc)
      zcat splits/${lang}.train.charunked.gz \
        | ${BPE_PATH}/learn_bpe.py -s ${num_merges} \
        > bpe/${lang}.bpe_${factor/./}_codes
      for setname in train dev test; do
        zcat splits/${lang}.${setname}.charunked.gz \
          | ${BPE_PATH}/apply_bpe.py -c bpe/${lang}.bpe_${factor/./}_codes \
          | gzip \
          > bpe/${lang}.${setname}.charunked.bpe_${factor/./}.gz
      done
    fi
  done
done

# Train KenLM on BPE (urk, UNK weirdness, don't trust this one)
if [ "$2" == "kenlm" ]; then
  mkdir -p kenlm
  for factor in ${BPE_FACTORS}; do
    for lang in ${LANGS_AND_REV}; do
      echo "KenLM for ${lang}"
      ${MOSES_PATH}/bin/lmplz --text <(zcat bpe/${lang}.train.charunked.bpe_${factor/./}.gz) -o 5 --arpa kenlm/${lang}.charunked.bpe_${factor/./}.arpa --discount_fallback
      ${MOSES_PATH}/bin/build_binary kenlm/${lang}.charunked.bpe_${factor/./}.arpa kenlm/${lang}.charunked.bpe_${factor/./}.bin
      rm kenlm/${lang}.charunked.bpe_${factor/./}.arpa
      zcat bpe/${lang}.test.charunked.bpe_${factor/./}.gz \
        | ${MOSES_PATH}/bin/query kenlm/${lang}.charunked.bpe_${factor/./}.bin \
        | sed 's/.*Total: //' \
        | cut -d ' ' -f 1 \
        | head -n -4 \
        > kenlm/${lang}.test.charunked.bpe_${factor/./}.kenlm.loglik
      rm kenlm/${lang}.charunked.bpe_${factor/./}.bin
    done

    echo ${LANGS} | sed -r -e 's/\s+/\t/g' \
      > kenlm/kenlm_charunked_bpe_${factor/./}_logliks.tsv
    paste $(for lang in ${LANGS}; do echo "kenlm/${lang}.test.charunked.bpe_${factor/./}.kenlm.loglik"; done) \
      >> kenlm/kenlm_charunked_bpe_${factor/./}_logliks.tsv
  done
fi

# Train my RNNLM on BPE
if [ "$2" == "rnnlmbpe" ]; then
  for factor in ${BPE_FACTORS}; do
    for lang in ${LANGS_AND_REV}; do
      echo "RNNLM for ${lang}-bpe_${factor/./}"
      # Make links so the RNNLM finds stuff (expanded because of dev -> valid name mismatch)
      mkdir -p rnnlm/inputdatalinks/${lang}_${factor/./}
      ln -sf ${MAINDIR}/splits/${lang}.train.charunked.gz rnnlm/inputdatalinks/${lang}_${factor/./}/train.txt.gz
      ln -sf ${MAINDIR}/splits/${lang}.dev.charunked.gz rnnlm/inputdatalinks/${lang}_${factor/./}/valid.txt.gz
      ln -sf ${MAINDIR}/splits/${lang}.test.charunked.gz rnnlm/inputdatalinks/${lang}_${factor/./}/test.txt.gz
      mkdir -p rnnlm/inputdatalinks/${lang}-bpe_${factor/./}
      ln -sf ${MAINDIR}/bpe/${lang}.bpe_${factor/./}_codes rnnlm/inputdatalinks/${lang}-bpe_${factor/./}/codes
      ln -sf ${MAINDIR}/bpe/${lang}.train.charunked.bpe_${factor/./}.gz rnnlm/inputdatalinks/${lang}-bpe_${factor/./}/train.txt.gz
      ln -sf ${MAINDIR}/bpe/${lang}.dev.charunked.bpe_${factor/./}.gz rnnlm/inputdatalinks/${lang}-bpe_${factor/./}/valid.txt.gz
      ln -sf ${MAINDIR}/bpe/${lang}.test.charunked.bpe_${factor/./}.gz rnnlm/inputdatalinks/${lang}-bpe_${factor/./}/test.txt.gz
      # Write the runner script
      MODELPREFIX="${lang}_bpe_${factor/./}_pelikan"
      cat <<- EOF > "rnnlm/${MODELPREFIX}.runner.sh"
#!/usr/bin/env bash
cd "${MAINDIR}/rnnlm"

# Training
$RNNLM_CALL \
  --data inputdatalinks/${lang}-bpe_${factor/./} \
  --dropouth 0.2 \
  --batch_size 40 \
  --vocab-size 999999 \
  --speller-mode none \
  --epochs 200 \
  --save ${MODELPREFIX}.pt \
  --boardcomment ${MODELPREFIX} \
  --no-histograms \
| tee ${MODELPREFIX}.log

# Getting scores to compare for tuning
$RNNLM_CALL \
  --data inputdatalinks/${lang}-bpe_${factor/./} \
  --vocab-size 999999 \
  --speller-mode none \
  --epochs 0 \
  --save ${MODELPREFIX}.pt \
| grep 'End of training' \
| tail -n 2 \
> ${MODELPREFIX}.outscores

# Getting actual logliks for regression
$RNNLM_CALL \
  --data inputdatalinks/${lang}-bpe_${factor/./} \
  --vocab-size 999999 \
  --speller-mode none \
  --epochs 0 \
  --save ${MODELPREFIX}.pt \
  --per-line \
| sed -n '/-----------------/,/Exiting now/p' \
| head -n -1 \
| tail -n +4 \
> ${MODELPREFIX}.outlogliks
EOF
      # Run training and testing code!
      # Due to the asyncronicity, merging has to be done later on :(
      ${RNNLM_RUNNER} "rnnlm/${MODELPREFIX}.runner.sh"
    done
  done
fi

# Train my RNNLM on chars
if [ "$2" == "rnnlmchar" ]; then
  for lang in ${LANGS_AND_REV}; do
    echo "RNNLM for ${lang}-char"
    # Make links so the RNNLM finds stuff (expanded because of dev -> valid name mismatch)
    mkdir -p rnnlm/inputdatalinks/${lang}
    ln -sf ${MAINDIR}/splits/${lang}.train.charunked.gz rnnlm/inputdatalinks/${lang}/train.txt.gz
    ln -sf ${MAINDIR}/splits/${lang}.dev.charunked.gz rnnlm/inputdatalinks/${lang}/valid.txt.gz
    ln -sf ${MAINDIR}/splits/${lang}.test.charunked.gz rnnlm/inputdatalinks/${lang}/test.txt.gz
    mkdir -p rnnlm/inputdatalinks/${lang}-char
    ln -sf ${MAINDIR}/splits/${lang}.train.charunked.chars.gz rnnlm/inputdatalinks/${lang}-char/train.txt.gz
    ln -sf ${MAINDIR}/splits/${lang}.dev.charunked.chars.gz rnnlm/inputdatalinks/${lang}-char/valid.txt.gz
    ln -sf ${MAINDIR}/splits/${lang}.test.charunked.chars.gz rnnlm/inputdatalinks/${lang}-char/test.txt.gz
    # Write the runner script
    MODELPREFIX="${lang}_chars_adam_ptb"
    cat <<- EOF > "rnnlm/${MODELPREFIX}.runner.sh"
#!/usr/bin/env bash
cd "${MAINDIR}/rnnlm"

# PTB style hyperparams
# Training
$RNNLM_CALL \
--data inputdatalinks/${lang}-char \
--save ${MODELPREFIX}.pt \
--boardcomment ${MODELPREFIX} \
--no-histograms \
--vocab-size 999999 \
--speller-mode none \
--epochs 500 \
--nlayers 3 \
--emsize 200 \
--nhid 1000 \
--alpha 0 \
--beta 0 \
--dropoute 0 \
--dropouth 0.25 \
--dropouti 0.1 \
--dropout 0.1 \
--wdrop 0.5 \
--wdecay 1.2e-6 \
--bptt 150 \
--batch_size 128 \
--optimizer adam \
--lr-lm 2e-3 \
--when 300 400 \
| tee ${MODELPREFIX}.log

# Getting scores to compare for tuning
$RNNLM_CALL \
--data inputdatalinks/${lang}-char \
--vocab-size 999999 \
--speller-mode none \
--epochs 0 \
--save ${MODELPREFIX}.pt \
| grep 'End of training' \
| tail -n 2 \
> ${MODELPREFIX}.outscores

# Getting actual logliks for regression
$RNNLM_CALL \
--data inputdatalinks/${lang}-char \
--vocab-size 999999 \
--speller-mode none \
--epochs 0 \
--save ${MODELPREFIX}.pt \
--per-line \
| sed -n '/-----------------/,/Exiting now/p' \
| head -n -1 \
| tail -n +4 \
> ${MODELPREFIX}.outlogliks
EOF
    # Run training and testing code!
    # Due to the asyncronicity, merging has to be done later on :(
    ${RNNLM_RUNNER} "rnnlm/${MODELPREFIX}.runner.sh"
  done
fi

# Training progess:
# for lang in bg cs da de el en es et fi fr hu it lt lv nl pl pt ro sk sl sv; do zcat rnnlm/inputdatalinks/${lang}-char/test.txt.gz | wc -w; echo -n "$lang "; grep 'end of epoch' rnnlm/${lang}_chars_adam_ptb.log  | sed 's/.*set) //;s/ |//' | tr '\n' ' '; echo ''; done
