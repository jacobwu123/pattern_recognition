"""Processing data tools for mp0.
"""
import re
import numpy as np
import collections

def title_cleanup(data):
    """Remove all characters except a-z, A-Z and spaces from the title,
       then convert all characters to lower case.

    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    """
    keys = data.keys()
    for key in keys:
        regex = re.compile('[^a-zA-Z ]')
        data[key][0] = regex.sub('',data[key][0]).lower()
    return


def most_frequent_words(data):
    """Find the more frequeny words (including all ties), returned in a list.

    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        max_words(list): List of strings containing the most frequent words.
    """
    max_words = []
    cnt = {}
    keys = data.keys()
    for key in keys:
        for word in data[key][0].split():
            if word not in cnt:
                cnt[word] = 0
            cnt[word] += 1

    values = list(cnt.values())
    values.sort()
    freq = values[len(values)-1]

    max_words.append(list(cnt.keys())[list(cnt.values()).index(freq)])

    return max_words


def most_positive_titles(data):
    """Computes the most positive titles.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        titles(list): List of strings containing the most positive titles,
                      include all ties.
    """
    titles = []
    pos = 0.0
    for title, score in data.values():
        if score > pos:
            pos = score

    for title, score in data.values():
        if score == pos:
            titles.append(title)

    return titles


def most_negative_titles(data):
    """Computes the most negative titles.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
     Returns:
        titles(list): List of strings containing the most negative titles,
                      include all ties.
    """
    titles = []
    pos = list(data.values())[0][1]
    for title, score in data.values():
        if score < pos:
            pos = score

    for title, score in data.values():
        if score == pos:
            titles.append(title)

    return titles


def compute_word_positivity(data):
    """Computes average word positivity.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        word_dict(dict): Key: word(str), value: word_index(int)
        word_avg(numpy.ndarray): numpy array where element
                                 #word_dict[word] is the
                                 average word positivity for word.
    """
    word_dict = {}
    word_avg = []
#    word_avg = word_score / word_count

    cnt_freq= {}
    cnt_score={}
    values = data.values()
    for title,score in values:
        for word in title.split():
            if word not in cnt_freq:
                cnt_freq[word] = 0
            cnt_freq[word] += 1
            if word not in cnt_score:
                cnt_score[word] = 0.0
            cnt_score[word] += score
            
    idx = 0
    words = cnt_score.keys();
    for word in words:
        word_dict[word] = idx
        word_avg.append(cnt_score[word]/cnt_freq[word]/1.0)
        idx += 1        



    return word_dict, word_avg


def most_postivie_words(word_dict, word_avg):
    """Computes the most positive words.
    Args:
        word_dict(dict): output from compute_word_positivity.
        word_avg(numpy.ndarray): output from compute_word_positivity.
    Returns:
        words(list):
    """
    words = []
    pos = 0.0

    keys = word_dict.keys()

    for key in keys:
        if word_avg[word_dict[key]] > pos:
            pos = word_avg[word_dict[key]]

    for key in keys:
        if word_dict[key] == word_avg.index(pos):
            words.append(key)
    print(words)

    return words


def most_negative_words(word_dict, word_avg):
    """Computes the most negative words.
    Args:
        word_dict(dict): output from compute_word_positivity.
        word_avg(numpy.ndarray): output from compute_word_positivity.
    Returns:
        words(list):
    """
    words = []

    pos = word_avg[0]

    keys = word_dict.keys()

    for key in keys:
        if word_avg[word_dict[key]] < pos:
            pos = word_avg[word_dict[key]]

    for key in keys:
        if word_dict[key] == word_avg.index(pos):
            words.append(key)
            
    print(words)

    return words
