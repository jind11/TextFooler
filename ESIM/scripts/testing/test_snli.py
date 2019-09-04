"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import time
import pickle
import argparse
import torch

from torch.utils.data import DataLoader
from esim.data import NLIDataset
from esim.model import ESIM
from esim.utils import correct_predictions


def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch['premise'].to(device)
            premises_lengths = batch['premise_length'].to(device)
            hypotheses = batch['hypothesis'].to(device)
            hypotheses_lengths = batch['hypothesis_length'].to(device)
            labels = batch['label'].to(device)

            _, probs = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)

            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy


def main(test_file, pretrained_file, batch_size=32):
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    checkpoint = torch.load(pretrained_file)

    # Retrieving model parameters from checkpoint.
    vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
    embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
    hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
    num_classes = checkpoint['model']['_classification.4.weight'].size(0)

    print("\t* Loading test data...")
    with open(test_file, 'rb') as pkl:
        test_data = NLIDataset(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint['model'])

    print(20 * "=",
          " Testing ESIM model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy = test(model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the ESIM model on\
 some dataset')
    parser.add_argument('test_data',
                        help="Path to a file containing preprocessed test data")
    parser.add_argument('checkpoint',
                        help="Path to a checkpoint with a pretrained model")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use during testing')
    args = parser.parse_args()

    main(args.test_data,
         args.checkpoint,
         args.batch_size)
