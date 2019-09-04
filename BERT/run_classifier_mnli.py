import os

command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/MNLI ' \
          '--bert_model bert-base-uncased ' \
          '--task_name mnli --output_dir results/MNLI --cache_dir pytorch_cache --do_eval --do_lower_case ' \
          '--do_resume'

os.system(command)