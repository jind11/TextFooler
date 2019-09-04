import os

command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/mr ' \
          '--bert_model bert-base-uncased ' \
          '--task_name mr --output_dir results/mr_retrain --cache_dir pytorch_cache --do_train --do_eval ' \
          '--do_lower_case '

os.system(command)