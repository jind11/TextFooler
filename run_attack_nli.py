import os

# for ESIM target model
# command = 'python attack_nli.py --dataset_path data/snli ' \
#           '--target_model esim --target_model_path ESIM/data/checkpoints/SNLI/best.pth.tar ' \
#           '--word_embeddings_path ESIM/data/preprocessed/SNLI/worddict.pkl ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache' \
#             '--output_dir results/snli_esim'

# for InferSent target model
command = 'python attack_nli.py --dataset_path data/snli ' \
          '--target_model infersent ' \
          '--target_model_path /scratch/jindi/adversary/BERT/results/SNLI ' \
          '--word_embeddings_path /data/medg/misc/jindi/nlp/embeddings/glove.840B/glove.840B.300d.txt ' \
          '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path /scratch/jindi/tf_cache ' \
          '--output_dir results/snli_infersent'

# for BERT target model
command = 'python attack_nli.py --dataset_path data/snli ' \
          '--target_model bert ' \
          '--target_model_path /scratch/jindi/adversary/BERT/results/SNLI ' \
          '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path /scratch/jindi/adversary/cos_sim_counter_fitting.npy ' \
          '--USE_cache_path /scratch/jindi/tf_cache ' \
          '--output_dir results/snli_bert'

os.system(command)