import os

command = 'python run_classifier.py --data_dir /afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/yelp ' \
          '--bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 ' \
          '--task_name yelp --output_dir results/yelp --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case ' \
          '--num_train_epochs 2.'

os.system(command)