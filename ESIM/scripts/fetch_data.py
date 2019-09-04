"""
Fetch datasets and pretrained word embeddings for the ESIM model.

By default, the script downloads the following.
    - The SNLI corpus;
    - GloVe word embeddings (840B - 300d).
"""
# Aurelien Coet, 2018.

import os
import argparse
import zipfile
import wget


def download(url, targetdir):
    """
    Download a file and save it in some target directory.

    Args:
        url: The url from which the file must be downloaded.
        targetdir: The path to the directory where the file must be saved.

    Returns:
        The path to the downloaded file.
    """
    print("* Downloading data from {}...".format(url))
    filepath = os.path.join(targetdir, url.split('/')[-1])
    wget.download(url, filepath)
    return filepath


def unzip(filepath):
    """
    Extract the data from a zipped file and delete the archive.

    Args:
        filepath: The path to the zipped file.
    """
    print("\n* Extracting: {}...".format(filepath))
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        for name in zf.namelist():
            # Ignore useless files in archives.
            if "__MACOSX" in name or\
               ".DS_Store" in name or\
               "Icon" in name:
                continue
            zf.extract(name, dirpath)
    # Delete the archive once the data has been extracted.
    os.remove(filepath)


def download_unzip(url, targetdir):
    """
    Download and unzip data from some url and save it in a target directory.

    Args:
        url: The url to download the data from.
        targetdir: The target directory in which to download and unzip the
                   data.
    """
    filepath = os.path.join(targetdir, url.split('/')[-1])

    if not os.path.exists(targetdir):
        print("* Creating target directory {}...".format(targetdir))
        os.makedirs(targetdir)

    # Download and unzip if the target directory is empty.
    if not os.listdir(targetdir):
        unzip(download(url, targetdir))
    # Skip downloading if the zipped data is already available.
    elif os.path.exists(filepath):
        print("* Found zipped data - skipping download...")
        unzip(filepath)
    # Skip download and unzipping if the unzipped data is already available.
    else:
        print("* Found unzipped data for {}, skipping download and unzip..."
              .format(targetdir))


if __name__ == "__main__":
    # Default data.
    snli_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    glove_url = "http://www-nlp.stanford.edu/data/glove.840B.300d.zip"

    parser = argparse.ArgumentParser(description='Download the SNLI dataset')
    parser.add_argument('--dataset_url',
                        default=snli_url,
                        help='URL of the dataset to download')
    parser.add_argument('--embeddings_url',
                        default=glove_url,
                        help='URL of the pretrained embeddings to download')
    parser.add_argument('--target_dir',
                        default=os.path.join('..', 'data'),
                        help='Path to a directory where data must be saved')
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    print(20*'=', "Fetching the dataset:", 20*'=')
    download_unzip(args.dataset_url, os.path.join(args.target_dir, "dataset"))

    print(20*'=', "Fetching the word embeddings:", 20*'=')
    download_unzip(args.embeddings_url,
                   os.path.join(args.target_dir, "embeddings"))
