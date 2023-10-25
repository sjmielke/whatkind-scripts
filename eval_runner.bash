#!/usr/bin/env bash
set -e

# How to use? Be in "/scratch/users/smielke2@jhu.edu/variance-and-variation/europarl.21" and call "bash eval_runner.bash marcc en 0.8 ../../ryans_english_file.txt".

MAINDIR="$(pwd)"

case "$1" in
  lanayru)
    BPE_PATH="/home/sjm/projects/ISI/subword-nmt/sennrich_bpe/"
    REVTOK_PY="/home/sjm/projects/JHU/variance-and-variation/reversible_tokenize.py"
    RNNLM_CALL="unbuffer python /home/sjm/projects/JHU/inlm/awd-lstm-lm/main.py --device cpu"
    RNNLM_RUNNER="bash"
    ;;

  clspgrid)
    BPE_PATH="/home/smielke/subword-nmt/"
    REVTOK_PY="/home/smielke/variance-and-variation/reversible_tokenize.py"
    RNNLM_CALL='CUDA_VISIBLE_DEVICES=`free-gpu` unbuffer python /home/smielke/inlm/awd-lstm-lm/main.py --device cuda'
    RNNLM_RUNNER='qsub -l hostname=c*,gpu=1 -q g.q -l mem_free=10G,ram_free=10G'
    ;;

  marcc)
    BPE_PATH="/scratch/users/smielke2@jhu.edu/subword-nmt"
    REVTOK_PY="/scratch/users/smielke2@jhu.edu/variance-and-variation/reversible_tokenize.py"
    RNNLM_CALL="ml singularity; TCLLIBPATH=/home-4/smielke2@jhu.edu/local/lib LC_ALL='C.UTF-8' unbuffer singularity exec --nv /scratch/users/smielke2\@jhu.edu/pytorch.0.4.1.simg python /scratch/users/smielke2\@jhu.edu/inlm/awd-lstm-lm/main.py --device cuda"
    RNNLM_RUNNER="sbatch -p gpuk80 --gres=gpu:1 --mem=50G --time=48:00:00"
    ;;

  *)
    echo $"Usage: $0 {lanayru|clspgrid|marcc} LANG FACTOR INPUTFILE"
    exit 1
esac

lang="$2"
factor="$3"
inputfile="$4"

# Make the tmp directory for this run
cd rnnlm
runname=$(mktemp -d eval_XXXXXX)
rundir="$(pwd)/${runname}"
echo "Made temporary dir ${rundir}."
cd ..

# Tokenize (other files already have this tokenization)
echo "Tokenizing ${inputfile} into ${rundir}/inputfile.revtok"
python3 "${REVTOK_PY}" --tok < "${inputfile}" > "${rundir}/inputfile.revtok"

# Char-UNK with ◊
echo "Char-UNKing ${rundir}/inputfile.revtok into ${rundir}/${lang}_${factor/./}/test.txt.gz."
python<<EOF
import collections
import gzip
with gzip.open("splits/${lang}.train.gz", 'rt', encoding = 'utf-8') as f:
  CHARSET = set([char for (char, count) in collections.Counter(f.read()).most_common() if count >= 25])
with open("${rundir}/inputfile.revtok", 'rt', encoding = 'utf-8') as f, gzip.open("${rundir}/testfile.charunked.gz", 'wt', encoding = 'utf-8') as fw:
    for line in f:
      fw.write(''.join(c if c in CHARSET or c.isspace() else '◊' for c in line))
EOF

if [ "$3" == "char" ]; then
  # Link training files
  cd "${rundir}"
  mkdir ${lang}
  ln -s ${MAINDIR}/splits/${lang}.train.charunked.gz       ${lang}/train.txt.gz
  ln -s ${MAINDIR}/splits/${lang}.dev.charunked.gz         ${lang}/valid.txt.gz
  mkdir ${lang}-char
  ln -s ${MAINDIR}/splits/${lang}.train.charunked.chars.gz ${lang}-char/train.txt.gz
  ln -s ${MAINDIR}/splits/${lang}.dev.charunked.chars.gz   ${lang}-char/valid.txt.gz
  cd ../..  # so inputfile will still resolve correctly, if it contains relative paths

  # Link file
  ln -s ${rundir}/testfile.charunked.gz ${rundir}/${lang}/test.txt.gz

  # Charify for char-level modeling
  echo "Charifying for ${lang}"
  python<<EOF
import gzip
SPACE = '⁀'  # This specific character is used in main.evaluate (L505) for the loss summing of characters into words
with gzip.open("${rundir}/testfile.charunked.gz", 'rt', encoding = 'utf-8') as f, \
    gzip.open("${rundir}/${lang}-char/test.txt.gz", 'wt', encoding = 'utf-8') as fw:
  for line in f:
    print(' '.join(SPACE.join(line.split())), file = fw)
EOF
  # Write the runner script
  cat <<- EOF > "${rundir}/${runname}.runner.sh"
#!/usr/bin/env bash
cd "${MAINDIR}/rnnlm"

# Getting actual logliks for regression

$RNNLM_CALL \
  --data ${rundir}/${lang}-char \
  --vocab-size 999999 \
  --speller-mode none \
  --epochs 0 \
  --save ${lang}_chars_adam_ptb_heinrich.pt \
  --per-line \
| sed -n '/-----------------/,/Exiting now/p' \
| head -n -1 \
| tail -n +4 \
> ${rundir}/${runname}.outlogliks

cp ${rundir}/${runname}.outlogliks ../"${inputfile}.chars_${2}.outlogliks"
EOF
else
  # Link training files
  cd "${rundir}"
  mkdir ${lang}_${factor/./}
  ln -s ${MAINDIR}/splits/${lang}.train.charunked.gz               ${lang}_${factor/./}/train.txt.gz
  ln -s ${MAINDIR}/splits/${lang}.dev.charunked.gz                 ${lang}_${factor/./}/valid.txt.gz
  mkdir ${lang}-bpe_${factor/./}
  ln -s ${MAINDIR}/bpe/${lang}.bpe_${factor/./}_codes              ${lang}-bpe_${factor/./}/codes
  ln -s ${MAINDIR}/bpe/${lang}.train.charunked.bpe_${factor/./}.gz ${lang}-bpe_${factor/./}/train.txt.gz
  ln -s ${MAINDIR}/bpe/${lang}.dev.charunked.bpe_${factor/./}.gz   ${lang}-bpe_${factor/./}/valid.txt.gz
  cd ../..  # so inputfile will still resolve correctly, if it contains relative paths

  # Link file
  ln -s testfile.charunked.gz ${lang}_${factor/./}/test.txt.gz

  # BPE-ize
  echo "BPE-ing ${rundir}/${lang}_${factor/./}/test.txt.gz using bpe/${lang}.bpe_${factor/./}_codes into ${rundir}/${lang}-bpe_${factor/./}/test.txt.gz"
  zcat ${rundir}/${lang}_${factor/./}/test.txt.gz \
    | ${BPE_PATH}/apply_bpe.py -c bpe/${lang}.bpe_${factor/./}_codes \
    | gzip \
    > ${rundir}/${lang}-bpe_${factor/./}/test.txt.gz

  # Write the runner script
  cat <<- EOF > "${rundir}/${runname}.runner.sh"
#!/usr/bin/env bash
cd "${MAINDIR}/rnnlm"

# Getting actual logliks for regression
$RNNLM_CALL \
  --data ${rundir}/${lang}-bpe_${factor/./} \
  --vocab-size 999999 \
  --speller-mode none \
  --epochs 0 \
  --save ${lang}_bpe_${factor/./}_pelikan.pt \
  --per-line \
| sed -n '/-----------------/,/Exiting now/p' \
| head -n -1 \
| tail -n +4 \
> ${rundir}/${runname}.outlogliks

cp ${rundir}/${runname}.outlogliks ../"${inputfile}.bpe_${factor/./}.outlogliks"
EOF
fi

# Start it!
${RNNLM_RUNNER} "${rundir}/${runname}.runner.sh"

echo "You'll find results in ${rundir} or directly at ${inputfile}.outlogliks !"
