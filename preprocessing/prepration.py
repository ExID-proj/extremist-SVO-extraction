import pandas as pd
import os
import spacy
import re
import numpy as np
import csv
import argparse


def split_dataframe(df, chunk_size):
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        yield df[i * chunk_size:(i + 1) * chunk_size].copy(deep=True)


def preprocess(df, path, filename, chunk_size=5000, verbose=True):
    if os.path.exists(path + '/%s_cleaned.csv'%(filename)):
        df = pd.read_csv(path + '/%s_cleaned.csv'%(filename))
        if verbose:
            print(df.shape, list(df), len(df.pid.unique()))
        return df

    def clean(s):
        WHITESPACEREGEX = r'[ \t\n\r\f\v]+'
        s.replace('\n', ' ')
        s.replace('\t', ' ')
        s = re.sub(WHITESPACEREGEX, ' ', s)
        return s.strip()

    def filterpunc(x):
        NONPUNCREGEX = r'[a-zA-Z0-9]'
        WHITESPACEREGEX = r'[ \t\n\r\f\v]+'
        if re.search(NONPUNCREGEX, x) is None:
            return ''
        x = re.sub(WHITESPACEREGEX, ' ', x)
        return x.strip()

    def filternonalpha(x):
        NONPUNCREGEX = r'[a-zA-Z]'
        WHITESPACEREGEX = r'[ \t\n\r\f\v]+'
        if re.search(NONPUNCREGEX, x) is None:
            return ''
        x = re.sub(WHITESPACEREGEX, ' ', x)
        return x.strip()

    def exclean(s):
        parantez = r'(\}|\{|\]|\[|\)|\()'
        symbols = r'(#|&)'
        DOTSREGEX = r'(\. |\.){2,}'
        DASHREGEX = r'(-( )?|_( )?){2,}'
        WHITESPACEREGEX = r'[ \t\n\r\f\v]+'
        s = re.sub(parantez, ' ', s)
        s = re.sub(symbols, ' ', s)
        s = re.sub(DOTSREGEX, '.', s)
        s = re.sub(DASHREGEX, '-', s)
        s = re.sub(WHITESPACEREGEX, ' ', s)
        return s.strip()

    # there should be a subject, verb, and object, so drop the sentences with less than 3 words
    def dropshortsen(df, min=3):
        def num_wrd(sen):
            return len(sen.split())

        df.loc[:, 'wrdC'] = df.sentence.apply(lambda x: num_wrd(x))
        temp = df.loc[df['wrdC'] >= min].copy()

        temp.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
        if verbose:
            print(df.shape, temp.shape)
        return temp

    df = df[['name_id', 'sentence']].copy()

    with open(path + filename + '_cleaned.csv', mode='w') as csv_file:
        fieldnames = ['id', 'pid', 'sentence']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        id = 1
        for subdf in split_dataframe(df, chunk_size=chunk_size):
            subdf['sentence'] = subdf.sentence.apply(lambda x: clean(x))

            # now split the paragraph to its the sentences i
            nlp = spacy.blank('en')
            nlp.add_pipe('sentencizer')
            def splitsen(x):
                return [sent.text for sent in nlp(x).sents]
            splited = pd.concat([pd.Series(row['name_id'], splitsen(row['sentence'])) for _, row in subdf.iterrows()]).reset_index()
            splited.rename(columns={'index': 'sentence', 0: 'pid'}, inplace=True)

            # detect rows which only contains punctuations, or only numbers
            splited['sentence'] = splited['sentence'].apply(lambda x: filterpunc(x))
            splited['sentence'] = splited['sentence'].apply(lambda x: filternonalpha(x))
            # drop empty rows
            splited['sentence'].replace('', np.nan, inplace=True)
            splited.dropna(subset=['sentence'], inplace=True)
            splited.reset_index(inplace=True)

            # removing several dots, or dashes, ...
            splited['sentence'] = splited['sentence'].apply(lambda x: exclean(x))
            splited = dropshortsen(splited)

            for i, row in splited.iterrows():
                writer.writerow({'id': id, 'pid': row['pid'], 'sentence': row['sentence']})
                id += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default='./data/',
                        type=str,
                        help="The input data dir. Should contain the .csv files for the task.")

    parser.add_argument("--datafile",
                        default='ISIS_corpus',
                        type=str,
                        help="name of the csv file we aim to extract the SVO triples.")

    args = parser.parse_args()

    df = pd.read_csv(args.data_dir+args.datafile+'.csv')
    preprocess(df, args.data_dir, args.datafile)


if __name__ == '__main__':
    main()