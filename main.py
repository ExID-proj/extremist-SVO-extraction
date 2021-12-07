import argparse
import pandas as pd
import json
import re

from find_SVOs import findSVOs, nlp
from utils import boolregex, _is_aux_verb, get_inout_grp, postproverb, mergesubtokens, get_inoutinstances


def readfile(path):
    try:
        df = pd.read_csv(path)
        return df
    except:
        print('file is not found')
        raise


def filter(sen, inoutlabels):
    for phrase in inoutlabels:
        if phrase in sen.lower().strip():
            return True
    return False


def cleaning(s):
    s = re.sub('“', '"', s)
    s = re.sub('”', '"', s)
    s = re.sub('‘', '', s)
    WHITESPACEREGEX = r'[ \t\n\r\f\v]+'
    s = re.sub(WHITESPACEREGEX, ' ', s)
    # because we merge the tokens such as non-whites, and we don't want to merge tokens such as washington - charging
    s = re.sub(r' - ', ' , ', s)

    return s.lower().strip()


def get_SVOs(df, inoutlabels, savepath):
    with open('%s/SVOs.json'%savepath, 'w') as out_file:
        for i, row in df.iterrows():
            sen = row['sentence'].lower().strip()

            triple = {'sentence': sen}

            tokens = nlp(sen)
            # merging the tokens such as non, -, white, as non-white
            try:
                mergesubtokens(tokens)
            except ValueError:
                raise

            # extracting the SVO triples - this return even the ones where subject of object are empty
            svos = findSVOs(tokens)

            # filtering out the SVOs we are interested in -
            # the ones where subject and object both are an instance of in- or out-group
            # dropping the ones with auxiliary verb
            # droppong those where subject and object are the same in- or out-group instance
            temp = []
            for item in svos:
                if boolregex(item[0], inoutlabels) and boolregex(item[2], inoutlabels):
                    if _is_aux_verb(nlp(item[1])) & postproverb(item[1]):

                        t = list(item)
                        t[0] = get_inout_grp(item[0], inoutlabels)
                        t[2] = get_inout_grp(item[2], inoutlabels)
                        if t[0] != t[2]:
                            temp.append(tuple(t))

            # dropping the redundant triples
            temp = list(set(temp))
            triple['extended_SVO'] = temp

            if triple['extended_SVO']:
                out_file.write(json.dumps(triple) + '\n')


def main(args):
    # read the main file containing sentences and the file which contains the instances of in- and out-groups
    maindf = readfile(args.data_dir + args.datafile + '.csv')
    inoutlabels = get_inoutinstances(args.data_dir + args.inoutfile + '.csv').group_name.tolist()

    # apply some minor cleaning on sentences
    maindf.sentence = maindf.sentence.apply(lambda x: cleaning(x))
    # filtering out the sentences which don't contain any of in- or out-group instances
    maindf = maindf[maindf['sentence'].apply(lambda x: filter(x, inoutlabels))]

    # call the function to extract the SVOs
    get_SVOs(maindf, inoutlabels, args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default='./data/',
                        type=str,
                        help="The input data dir. Should contain the .csv files for the task.")

    parser.add_argument("--datafile",
                        default='NSM_corpus_cleaned',
                        type=str,
                        help="name of the csv file we aim to extract the SVO triples.")

    parser.add_argument("--inoutfile",
                        default='NSM_ingroups_outgroups',
                        type=str,
                        help="name of the csv file which contains in- and out-group instances.")

    parser.add_argument("--save_dir",
                        default='./save/',
                        type=str,
                        help="path to saving directory.")

    args = parser.parse_args()
    main(args)
