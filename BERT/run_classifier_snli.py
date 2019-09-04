import os

command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/SNLI/snli_1.0 ' \
          '--bert_model bert-base-uncased ' \
          '--task_name snli --output_dir results/SNLI_retrain --cache_dir pytorch_cache  --do_train --do_eval --do_lower_case ' \
          # '--do_resume'

os.system(command)