# TextFooler
A Model for Natural Language Attack on Text Classification and Inference

This is the source code for the paper: [Jin, Di, et al. "Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment." arXiv preprint arXiv:1907.11932 (2019)](https://arxiv.org/pdf/1907.11932.pdf).

## Prerequisites:
* Pytorch >= 0.4
* Tensorflow >= 1.0 
* Numpy
* Python >= 3.6

## How to use

* Run the following code to install the **esim** package:

 ```
cd ESIM
python setup.py install
cd ..
```

* (Optional) Run the following code to pre-compute the cosine similarity scores between word pairs based on the [counter-fitting word embeddings](https://github.com/nmrksic/counter-fitting).

```
python comp_cos_sim_mat.py [PATH_TO_COUNTER_FITTING_WORD_EMBEDDINGS]
```

* Run the following code to generate the adversaries for text classification:

```
python attack_classification.py
```

For Natural langauge inference:

```
python attack_nli.py
```

Examples of run code for these two files are in [run_attack_classification.py]() and [run_attack_nli.py](). Here we explain each required argument in details:

  * --dataset_path: The path to the dataset. We put the 1000 examples for each dataset we used in the paper in the folder [data]().
  * --target_model: Name of the target model such as ''bert''.
  * --target_model_path: The path to the trained parameters of the target model.
  * --counter_fitting_embeddings_path: The path to the counter-fitting word embeddings.
  * --counter_fitting_cos_sim_path: This is optional. If given, then the pre-computed cosine similarity scores based on the counter-fitting word embeddings will be loaded to save time. If not, it will be calculated.
  * --USE_cache_path: The path to save the USE model file (Downloading is automatic if this path is empty).
  
