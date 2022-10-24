# Data Processing
Prepare pre-training and fine-tuning data.


## Pre-training
In this section, you learn how to download the raw Bookcorpus and English Wikipedia data, split them into training and 
validation sets, tokenize and binarize them to the format required by [fairseq](https://github.com/facebookresearch/fairseq). 

**Note**: We can't access to the original Wikipedia and Bookcorpus data trained for BERT. This is a reproduced version. 
* Download English Wikipedia from 
  [google drive]( https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT?usp=sharing) 
  (refer to [issue](https://github.com/mlcommons/training/issues/377)) and extract:
  ```bash
  pip install gdown
  gdown --id "18K1rrNJ_0lSR9bsLaoP3PkQeSFO-9LE7"
  bzip2 -dk enwiki-latest-pages-articles.xml.bz2
  ```
* Download Bookcorpus from [the Pile](https://github.com/EleutherAI/the-pile) and extract:
  ```bash
  wget https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz
  tar xvzf books1.tar.gz  # all books (text files) saved in the books1/epubtxt folder
  ```
* Clean Wikipedia and Bookcorpus (borrowed from 
  [24hBERT](https://github.com/IntelLabs/academic-budget-bert/tree/main/dataset#data-processing)):
  ```bash
  git clone https://github.com/IntelLabs/academic-budget-bert.git
  cd academic-budget-bert
  pip install -r requirements.txt
  cd dataset
  # generate wiki_one_article_per_line.txt (13G)
  python process_data.py -f <path_to_xml> -o <clean_dir> --type wiki  
  # generate bookcorpus_one_article_per_line.txt (6.2G)
  python process_data.py -f <path_to_text_files> -o <clean_dir> --type bookcorpus 
  ```
* Split data into training and validation sets (borrowed from 
  [24hBERT](https://github.com/IntelLabs/academic-budget-bert/tree/main/dataset#initial-sharding)):
  ```bash
  python shard_data.py \
    --dir <clean_dir> \
    -o <split_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.15
  
  # merge all training files into one single file, same for test files
  python merge_shards.py \
    --data <clean_dir> \
    --output_dir <merge_dir> \
    --grep training \ 
    --ratio 256
  
  python merge_shards.py \
    --data <clean_dir> \
    --output_dir <merge_dir> \
    --grep test \ 
    --ratio 128
  
  # rename in the format required by fairseq
  cp <merge_dir>/training0.txt <merge_dir>/train.txt
  cp <merge_dir>/test0.txt <merge_dir>/valid.txt
  rm <merge_dir>/training0.txt <merge_dir>/test0.txt
  ```
  We use ```--frac_test 0.15``` to make sure the size of the training set is 16G for fair comparison to the 
  original BERT training data. If the validation set (3G) is too large for you, you can delete some test files randomly.
  
  
* Tokenize with GPT-2/RoBERTa tokenizer and binarize the bpe data (required by fairseq):
  ```bash
  wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
  wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
  wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
  
  cd <path_to_fairseq>
  for SPLIT in train valid; do \
    python examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs <merge_dir>/${SPLIT}.txt \
        --outputs <bpe_dir>/${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
  done
  
  fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref <bpe_dir>/train.bpe \
    --validpref <bpe_dir>/valid.bpe \
    --destdir <bin_dir> \
    --workers 60
  ```
  
## Fine-tuning
We borrow the scripts from 
[fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.glue.md).
* Download GLUE data:
  ```bash
  wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
  python download_glue_data.py --data_dir <glue_dir> --tasks all
  ```
* Tokenize and binarize:
  ```bash
  cd <path_to_fairseq>
  bash examples/roberta/preprocess_GLUE_tasks.sh <glue_dir> ALL
  ```


  
