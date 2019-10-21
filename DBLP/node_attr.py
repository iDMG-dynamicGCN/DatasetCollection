import sqlite3  # save dblp to sqlite3 later
from tqdm import tqdm
import numpy as np
import pdb
import re
# import numpy as np 
from collections import defaultdict
import os
import sys


#    # author id:name 

# retrive from gallery

year = range(2010, 2020)
author = np.load('author_dict.npy').item()
# print((author))

def generate_author_dict(name=None):
    # read authors
    author_file='top500-author.txt'

    author_dict = defaultdict(int)

    with open(author_file) as f:
        authors = f.readlines()
        for i in range(authors.__len__()):
            author_dict[authors[i].strip()]={'idx': i, 'title': []}  # author_name : id

    np.save('author_dict', author_dict)
# generate_author_dict()

def generate_dataset():

    # read dataset    
    dblp = 'idx-year-title-authors.txt'
    # dblp = 'test.txt'
    author_title = defaultdict(str)
    with open(dblp) as f:
        # handle double line 
        index = dict.fromkeys(['year', 'title', 'author'])
        dataset = {}
        for line in tqdm(f):
            if not line:
                break
            line = line.split('\t')
            if len(line) != 4:
                # print(line)
                continue
            else:
                if int(line[1]) not in year:
                    continue
                i = index.copy()
                i['year'] = int(line[1])
                i['title'] = re.findall('[a-zA-Z]{3,}',line[2][1:-1])
                # i['author'] = line[3].strip(',\n').split(' ,')
                i['author'] = line[3].split(', ')[:-1]
                dataset[line[0]] = i

        # print(dataset)
        np.save('idx-year-title-author.npy', dataset)
        

    # pdb.set_trace()

def extract_attr_from_dataset(name):
    dataset = np.load(name).item()
    for idx, value in tqdm(dataset.items()):
        if value['year'] not in year:
            continue
        for _author in value['author']:
            if _author in author.keys():
                author[_author]['title'].extend(value['title'])

    for _author in author.values():
        _author['title'] = set(_author['title'])
    np.save('extracted_token', author)

# generate_dataset()
# extract_attr_from_dataset('idx-year-title-author.npy')

def clean_data(name):
    author_info = np.load(name).item()
    for value in tqdm(author_info.values()):
        value['title'] = clean_rule(value['title'])
    np.save('extracted_token_filtered.npy', author_info)


def clean_rule(title_list):
    title_list = [title_token.lower() for title_token in title_list]
    # print(title_list)
    return title_list

clean_data('extracted_token.npy')

def merge_title_to_matrix(name='extracted_token_filtered.npy'):
    title_all_token = []
    author_dict = np.load(name).item()
    for value in tqdm(author_dict.values()):
        title_all_token.extend(value['title'])
    title_all_token = list(set(title_all_token))
    title_all_token.sort()
    print(title_all_token[-100:])
    print(f'token_len: {title_all_token.__len__()}')
    pass

merge_title_to_matrix()

