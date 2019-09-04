import os

command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/imdb ' \
          '--bert_model bert-base-uncased --max_seq_length 256 --train_batch_size 32 ' \
          '--task_name imdb --output_dir results/imdb --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case ' \
          '--num_train_epochs 3.'

os.system(command)