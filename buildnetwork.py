import pandas as pd
import json
import argparse
import spacy
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from utils import get_inoutinstances

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])


def read_SVOs(pathtojason):
    for line in open(pathtojason, 'r'):
        yield json.loads(line)


def getverbroot(verb):
    prefix = ''
    if '!' in verb:
        prefix = '!'
        verb = verb.replace('!', '')
    lemma = [token.lemma_ for token in nlp(verb)][0]
    return prefix + lemma


def getnodes(pathtojason, pathtoinoutfile, savedir):
    inout_file = get_inoutinstances(pathtoinoutfile)

    def inout_search(subject):
        for index, row in inout_file.iterrows():
            if subject.lower().strip() == row['group_name'].lower().strip():
                if row['group_type'] == '':
                    raise
                return row['group_type']
        raise

    nodesdf = pd.DataFrame(columns=['name', 'type'])
    for row in read_SVOs(pathtojason):
        svlist = row['extended_SVO']
        for sv in svlist:
            nodesdf = nodesdf.append({'name': sv[0], 'type': inout_search(sv[0])}, ignore_index=True)
            nodesdf = nodesdf.append({'name': getverbroot(sv[1]), 'type': 'verb'}, ignore_index=True)
            nodesdf = nodesdf.append({'name': sv[2], 'type': inout_search(sv[2])}, ignore_index=True)

    nodesdf.drop_duplicates(subset=['name', 'type'], keep='first', inplace=True)
    nodesdf['Id'] = range(0, nodesdf.shape[0])
    nodesdf.to_csv('./%s/nodes.csv'%savedir, index=False)

    nodes_subobject = nodesdf.loc[(nodesdf.type == 'ingroup') | (nodesdf.type == 'outgroup')]
    nodes_verb = nodesdf.loc[nodesdf.type == 'verb']

    return nodesdf, nodes_verb, nodes_subobject


def getedges(Tnodesubjobjdict, Tnodeverbdict, nodes_subobject, pathtojason, savedir):
    edgedic_in = {}
    edgedic_out = {}
    for row in read_SVOs(pathtojason):
        svlist = row['extended_SVO']
        for sv in svlist:
            verb = getverbroot(sv[1])
            subject = sv[0]
            object = sv[2]

            verbid = Tnodeverbdict[verb]
            subjectid = Tnodesubjobjdict[subject]
            objectid = Tnodesubjobjdict[object]

            subjecttype = nodes_subobject.loc[nodes_subobject.Id == subjectid]['type'].tolist()
            assert len(subjecttype) == 1

            edge1 = str(subjectid) + '_' + str(verbid)
            edge2 = str(verbid) + '_' + str(objectid)

            if subjecttype[0] == 'ingroup':
                if edge1 in edgedic_in:
                    edgedic_in[edge1] += 1
                else:
                    edgedic_in[edge1] = 1
                if edge2 in edgedic_in:
                    edgedic_in[edge2] += 1
                else:
                    edgedic_in[edge2] = 1
            elif subjecttype[0] == 'outgroup':
                if edge1 in edgedic_out:
                    edgedic_out[edge1] += 1
                else:
                    edgedic_out[edge1] = 1
                if edge2 in edgedic_out:
                    edgedic_out[edge2] += 1
                else:
                    edgedic_out[edge2] = 1
            else:
                raise

    edgesdf_in = pd.DataFrame(columns=['source', 'target', 'weight'])
    for key, value in edgedic_in.items():
        source, target = key.split('_')
        weight = int(value)
        edgesdf_in = edgesdf_in.append({'source': int(source), 'target': int(target), 'weight': weight},
                                       ignore_index=True)
    edgesdf_in.to_csv('./%s/edges_In.csv'%savedir, index=False)

    edgesdf_out = pd.DataFrame(columns=['source', 'target', 'weight'])
    for key, value in edgedic_out.items():
        source, target = key.split('_')
        weight = int(value)
        edgesdf_out = edgesdf_out.append({'source': int(source), 'target': int(target), 'weight': weight},
                                         ignore_index=True)
    edgesdf_out.to_csv('./%s/edges_Out.csv'%savedir, index=False)

    return edgesdf_in, edgesdf_out


def testnetwork(nodes, edgesdf_in, edgesdf_out, pathtojason):
    totaledge = 0
    for row in read_SVOs(pathtojason):
        svolist = row['extended_SVO']
        totaledge += len(svolist)

    assert totaledge * 2 == edgesdf_in['weight'].sum() + edgesdf_out['weight'].sum()

    for index, row in edgesdf_in.iterrows():
        type1 = nodes.loc[nodes.Id == row['source']]['type'].tolist()[0]
        if not (type1 == 'ingroup' or type1 == 'verb'):
            print(index, row)

    for index, row in edgesdf_out.iterrows():
        type1 = nodes.loc[nodes.Id == row['source']]['type'].tolist()[0]

        if not (type1 == 'outgroup' or type1 == 'verb'):
            print(index, row)


def main(args):
    nodesdf, nodes_verb, nodes_subobject = getnodes(args.json_dir + args.jsonfile + '.json',
                                                    args.data_dir + args.inoutfile + '.csv', args.save_dir)

    nodeverbdict = dict(nodes_verb[['Id', 'name']].values)
    nodesubjobjdict = dict(nodes_subobject[['Id', 'name']].values)

    Tnodesubjobjdict = {value: key for key, value in nodesubjobjdict.items()}
    Tnodeverbdict = {value: key for key, value in nodeverbdict.items()}

    edgesdf_in, edgesdf_out = getedges(Tnodesubjobjdict, Tnodeverbdict, nodes_subobject,
                                       args.json_dir + args.jsonfile + '.json', args.save_dir)

    testnetwork(nodesdf, edgesdf_in, edgesdf_out, args.json_dir + args.jsonfile + '.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default='./data/',
                        type=str,
                        help="")

    parser.add_argument("--json_dir",
                        default='./save/',
                        type=str,
                        help="path to .json SVO file")

    parser.add_argument("--jsonfile",
                        default='SVOs',
                        type=str,
                        help="name of the json file which contains the SVO triples.")

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
