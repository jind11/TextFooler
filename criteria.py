from __future__ import division
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from pattern.en import conjugate, lemma, lexeme, PRESENT, SG, PL, PAST, PROGRESSIVE
import random


# Function 0: List of stop words
def get_stopwords():
    '''
    :return: a set of 266 stop words from nltk. eg. {'someone', 'anyhow', 'almost', 'none', 'mostly', 'around', 'being', 'fifteen', 'moreover', 'whoever', 'further', 'not', 'side', 'keep', 'does', 'regarding', 'until', 'across', 'during', 'nothing', 'of', 'we', 'eleven', 'say', 'between', 'upon', 'whole', 'in', 'nowhere', 'show', 'forty', 'hers', 'may', 'who', 'onto', 'amount', 'you', 'yours', 'his', 'than', 'it', 'last', 'up', 'ca', 'should', 'hereafter', 'others', 'would', 'an', 'all', 'if', 'otherwise', 'somehow', 'due', 'my', 'as', 'since', 'they', 'therein', 'together', 'hereupon', 'go', 'throughout', 'well', 'first', 'thence', 'yet', 'were', 'neither', 'too', 'whether', 'call', 'a', 'without', 'anyway', 'me', 'made', 'the', 'whom', 'but', 'and', 'nor', 'although', 'nine', 'whose', 'becomes', 'everywhere', 'front', 'thereby', 'both', 'will', 'move', 'every', 'whence', 'used', 'therefore', 'anyone', 'into', 'meanwhile', 'perhaps', 'became', 'same', 'something', 'very', 'where', 'besides', 'own', 'whereby', 'whither', 'quite', 'wherever', 'why', 'latter', 'down', 'she', 'sometimes', 'about', 'sometime', 'eight', 'ever', 'towards', 'however', 'noone', 'three', 'top', 'can', 'or', 'did', 'seemed', 'that', 'because', 'please', 'whereafter', 'mine', 'one', 'us', 'within', 'themselves', 'only', 'must', 'whereas', 'namely', 'really', 'yourselves', 'against', 'thus', 'thru', 'over', 'some', 'four', 'her', 'just', 'two', 'whenever', 'seeming', 'five', 'him', 'using', 'while', 'already', 'alone', 'been', 'done', 'is', 'our', 'rather', 'afterwards', 'for', 'back', 'third', 'himself', 'put', 'there', 'under', 'hereby', 'among', 'anywhere', 'at', 'twelve', 'was', 'more', 'doing', 'become', 'name', 'see', 'cannot', 'once', 'thereafter', 'ours', 'part', 'below', 'various', 'next', 'herein', 'also', 'above', 'beside', 'another', 'had', 'has', 'to', 'could', 'least', 'though', 'your', 'ten', 'many', 'other', 'from', 'get', 'which', 'with', 'latterly', 'now', 'never', 'most', 'so', 'yourself', 'amongst', 'whatever', 'whereupon', 'their', 'serious', 'make', 'seem', 'often', 'on', 'seems', 'any', 'hence', 'herself', 'myself', 'be', 'either', 'somewhere', 'before', 'twenty', 'here', 'beyond', 'this', 'else', 'nevertheless', 'its', 'he', 'except', 'when', 'again', 'thereupon', 'after', 'through', 'ourselves', 'along', 'former', 'give', 'enough', 'them', 'behind', 'itself', 'wherein', 'always', 'such', 'several', 'these', 'everyone', 'toward', 'have', 'nobody', 'elsewhere', 'empty', 'few', 'six', 'formerly', 'do', 'no', 'then', 'unless', 'what', 'how', 'even', 'i', 'indeed', 'still', 'might', 'off', 'those', 'via', 'fifty', 'each', 'out', 'less', 're', 'take', 'by', 'hundred', 'much', 'anything', 'becoming', 'am', 'everything', 'per', 'full', 'sixty', 'are', 'bottom', 'beforehand'}
    '''
    stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
    stop_words = set(stop_words)
    return stop_words
    # StopWords = {}
    # StopWords['nltk'] = set(nltk.corpus.stopwords.words('english'))
    #
    # import spacy
    # nlp = spacy.load("en")
    # StopWords['spacy'] = nlp.Defaults.stop_words
    #
    # return StopWords['nltk']  # | StopWords['spacy']


UniversalPos = ['NOUN', 'VERB', 'ADJ', 'ADV',
                'PRON', 'DET', 'ADP', 'NUM',
                'CONJ', 'PRT', '.', 'X']


# Function 1:
def get_pos(sent, tagset='universal'):
    '''
    :param sent: list of word strings
    tagset: {'universal', 'default'}
    :return: list of pos tags.
    Universal (Coarse) Pos tags has  12 categories
        - NOUN (nouns)
        - VERB (verbs)
        - ADJ (adjectives)
        - ADV (adverbs)
        - PRON (pronouns)
        - DET (determiners and articles)
        - ADP (prepositions and postpositions)
        - NUM (numerals)
        - CONJ (conjunctions)
        - PRT (particles)
        - . (punctuation marks)
        - X (a catch-all for other categories such as abbreviations or foreign words)
    '''
    if tagset == 'default':
        word_n_pos_list = nltk.pos_tag(sent)
    elif tagset == 'universal':
        word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
    _, pos_list = zip(*word_n_pos_list)
    return pos_list


# Function 2: Pos Filter
def pos_filter(ori_pos, new_pos_list):
    same = [True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['NOUN', 'VERB']))
            else False
            for new_pos in new_pos_list]
    return same


# Function 3:
def get_v_tense(sent):
    '''
    :param sent: a list of words
    :return tenses: a dict {key (word ix): value (tense, e.g. VBD)}
    pos of verbs
        - VB	Verb, base form
        - VBD	Verb, past tense
        - VBG	Verb, gerund or present participle
        - VBN	Verb, past participle
        - VBP	Verb, non-3rd person singular present
        - VBZ	Verb, 3rd person singular present
    '''
    word_n_pos_list = nltk.pos_tag(sent)
    _, pos_list = zip(*word_n_pos_list)
    tenses = {w_ix: tense for w_ix, tense in enumerate(pos_list) if tense.startswith('V')}
    return tenses


def change_tense(word, tense, lemmatize=False):
    '''
    en.verb.tenses():
        ['past', '3rd singular present', 'past participle', 'infinitive',
         'present participle', '1st singular present', '1st singular past',
         'past plural', '2nd singular present', '2nd singular past',
         '3rd singular past', 'present plural']
    :return:
    reference link: https://www.clips.uantwerpen.be/pages/pattern-en#conjugation
    '''
    if lemmatize:
        word = WordNetLemmatizer().lemmatize(word, 'v')
        # if pos(word) is not verb, return word
    lookup = {
        'VB': conjugate(verb=word, tense=PRESENT, number=SG),
        'VBD': conjugate(verb=word, tense=PAST, aspect=PROGRESSIVE, number=SG),
        'VBG': conjugate(verb=word, tense=PRESENT, aspect=PROGRESSIVE, number=SG),
        'VBN': conjugate(verb=word, tense=PAST, aspect=PROGRESSIVE, number=SG),
        'VBP': conjugate(verb=word, tense=PRESENT, number=PL),
        'VBZ': conjugate(verb=word, tense=PRESENT, number=SG),
    }
    return lookup[tense]


def get_sent_list():
    file_format = "/afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/{}/test_lm.txt"
    content = []
    for dataset in ['ag', 'fake', 'mr', 'yelp']:
        file = file_format.format(dataset)
        with open(file) as f:
            content += [line.strip().split() for line in f if line.strip()]
    return content


def check_pos(sent_list, win_size=10):
    '''
    :param sent_list:
    :param win_size:
    :param pad_size:
    :return: diff_ix = Counter({0: 606, 1: 180, 2: 42, 3: 15, 4: 5, 5: 1})
    len(sent_list) = 60139
    '''

    sent_list = sent_list[:]
    random.shuffle(sent_list)
    sent_list = sent_list[:100]

    center_ix = [random.randint(0 + win_size // 2, len(sent) - 1 - win_size // 2)
                 if len(sent) > win_size else len(sent) // 2
                 for sent in sent_list]
    word_range = [[max(0, cen_ix - win_size // 2), min(len(sent), cen_ix + win_size // 2)]
                  for cen_ix, sent in zip(center_ix, sent_list)]

    assert len(center_ix) == len(word_range)
    assert len(center_ix) == len(sent_list)

    corr_pos = [get_pos(sent)[word_range[sent_ix][0]: word_range[sent_ix][1]] for sent_ix, sent in enumerate(sent_list)]
    part_pos = [get_pos(sent[word_range[sent_ix][0]: word_range[sent_ix][1]]) for sent_ix, sent in enumerate(sent_list)]
    # corr_pos = [sent_pos[pad_size: -pad_size] if len(sent_pos) > 2 * pad_size else sent_pos
    #             for sent_ix, sent_pos in enumerate(corr_pos)]
    # part_pos = [sent_pos[pad_size: -pad_size] if len(sent_pos) > 2 * pad_size else sent_pos
    #             for sent_ix, sent_pos in enumerate(part_pos)]

    diff_ix = []
    diff_s_ix = []
    for sent_ix, (sent_pos_corr, sent_pos_part) in enumerate(zip(corr_pos, part_pos)):
        cen_ix = center_ix[sent_ix] - word_range[sent_ix][0]
        if sent_pos_corr[cen_ix] != sent_pos_part[cen_ix]:
            diff_s_ix += [sent_ix]
    # show_var(["diff_s_ix", "win_size"])

    if diff_s_ix:
        import pdb;
        pdb.set_trace()
    # if sent_pos_corr != sent_pos_part:
    #     diff_ix += [w_ix for w_ix, (p_corr, p_part) in enumerate(zip(sent_pos_corr, sent_pos_part))
    #                 if p_corr != p_part]
    #     diff_s_ix += [sent_ix]


def main():
    # Function 0:
    stop_words = get_stopwords()

    # Function 1:
    sent = 'i have a dream'.split()
    pos_list = get_pos(sent)
    sent_list = get_sent_list()
    for _ in range(10):
        check_pos(sent_list)
    import pdb;
    pdb.set_trace()

    # Function 2:
    ori_pos = 'NOUN'
    new_pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV', 'X', '.']
    same = pos_filter(ori_pos, new_pos_list)

    # Function 3:
    tenses = get_v_tense(sent)

    # this following one does not work, due to the failure to import
    # NodeBox English linguistic library (http://nodebox.net/code/index.php/Linguistics)
    new_word = change_tense('made', 'VBD')
    import pdb;
    pdb.set_trace()


if __name__ == "__main__":
    main()
