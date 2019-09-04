import os

command = 'python run_classifier.py --data_dir /afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/fake ' \
          '--bert_model bert-base-uncased --max_seq_length 256 --train_batch_size 16 ' \
          '--task_name fake --output_dir results/fake --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case '

os.system(command)