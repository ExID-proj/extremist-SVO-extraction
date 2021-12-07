from spacy.matcher import Matcher
import spacy
import re
import pandas as pd


nlp = spacy.load('en_core_web_lg')
matcher1 = Matcher(nlp.vocab)
matcher2 = Matcher(nlp.vocab)

passive_rule_0 = [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '?'}, {'DEP': 'neg', 'OP': '?'},
                  {'DEP': 'prep', 'OP': '?'}, {'DEP': 'poss', 'OP': '?'}, {'DEP': 'amod', 'OP': '?'},
                  {'DEP': 'det', 'OP': '?'}, {'DEP': 'pobj', 'OP': '?'}, {'DEP': 'auxpass'}, {'DEP': 'neg', 'OP': '?'},
                  {'DEP': 'advmod', 'OP': '?'}, {'DEP': 'preconj', 'OP': '?'}, {'DEP': 'dep', 'OP': '?'},
                  {'TAG': 'VBN'}]
passive_rule_1 = [{'DEP': 'nsubjpass'}, {'DEP': 'auxpass'}, {'DEP': 'neg', 'OP': '?'}, {'DEP': 'advmod', 'OP': '*'},
                  {'DEP': 'dep', 'OP': '?'}, {'TAG': 'VBN'}]
passive_rule_2 = [{'DEP': 'nsubjpass'}, {'DEP': 'auxpass'}, {'DEP': 'neg', 'OP': '?'}, {'DEP': 'dep', 'OP': '?'},
                  {'DEP': 'advmod', 'OP': '*'}, {'TAG': 'VBN'}]
passive_rule_3 = [{'DEP': 'nsubjpass'}, {'DEP': 'auxpass'}, {'DEP': 'advmod', 'OP': '?'}, {'DEP': 'neg', 'OP': '?'},
                  {'DEP': 'dep', 'OP': '?'}, {'TAG': 'VBN'}]
passive_rule_4 = [{'DEP': 'auxpass'}, {'DEP': 'neg', 'OP': '?'}, {'DEP': 'advmod', 'OP': '*'},
                  {'DEP': 'dep', 'OP': '?'}, {'TAG': 'VBN'}]
passive_rule_5 = [{'DEP': 'auxpass'}, {'DEP': 'poss', 'OP': '?'}, {'DEP': 'nsubj'}, {'TAG': 'VBN'}]
passive_rule_6 = [{'DEP': 'auxpass'}, {'DEP': 'nsubjpass'}, {'TAG': 'VBN'}]


merge_rule = [{'IS_ALPHA': True, 'IS_SPACE': False},
           {'ORTH': '-'},
           {'IS_ALPHA': True, 'IS_SPACE': False}]


def get_inoutinstances(pathtoinoutfile):
    df = pd.read_csv(pathtoinoutfile)
    df['group_name'] = df['group_name'].apply(lambda x: str(x).lstrip().rstrip().lower())
    df['group_type'] = df['group_type'].apply(lambda x: str(x).lstrip().rstrip().lower())
    df['temp1'] = df['group_name'].apply(lambda x: len(x.split()))
    df['temp2'] = df['group_name'].apply(lambda x: len(x))
    df = df.sort_values(by=['temp1','temp2'], ascending=(False, False))
    return df[['group_name', 'group_type']]


def is_in_matches(m_indexes, vid):
    for m_index in m_indexes:
        if m_index[0] <= vid <= m_index[1]:
            return True
    return False


def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


def passive_phrases(tokens):
    def widest_match(matches):
        final_matches = []
        for i in range(0, len(matches)):
            f_match = matches[i]
            flag = True
            for j in range(0, len(matches)):
                if i != j:
                    s_match = matches[j]

                    if f_match[1] >= s_match[1] and f_match[2] <= s_match[2]:
                        flag = False
                        break
            if flag:
                final_matches.append(f_match)
        return final_matches

    p_phrases = []
    m_indexes = []

    if not _is_passive(tokens):
        return False, p_phrases, m_indexes

    matcher1.add('Passive', [passive_rule_0, passive_rule_1, passive_rule_2, passive_rule_3, passive_rule_4,
                            passive_rule_5, passive_rule_6])
    matches = matcher1(tokens)

    if len(matches) > 0:
        f_matches = widest_match(matches)
        for match in f_matches:
            p_phrases.append(tokens[match[1]:match[2]])
            m_indexes.append((match[1], match[2]))
        return True, p_phrases, m_indexes

    else:
        return False, p_phrases, m_indexes


def mergesubtokens(tokens):
    matcher2.add("Hyphenated", [merge_rule])
    matches = matcher2(tokens)
    prev_st = 0
    prev_ed = 0
    with tokens.retokenize() as retokenizer:
        if matches:
            for match_id, start, end in matches:
                if prev_ed == 0:
                    prev_st = start
                    prev_ed = end

                elif start == prev_ed - 1:
                    prev_ed = end

                elif start != prev_ed - 1:
                    retokenizer.merge(tokens[prev_st:prev_ed])
                    prev_ed = end
                    prev_st = start

            retokenizer.merge(tokens[prev_st:prev_ed])


def boolregex(phrase, inoutlabels):
    def clean(s):
        symbols = r'(#|&|:|"|\?)'
        WHITESPACEREGEX = r'[ \t\n\r\f\v]+'
        s = re.sub(symbols, ' ', s)
        s = re.sub(WHITESPACEREGEX, ' ', s)
        return s.strip()

    phrase = clean(phrase)
    for form in inoutlabels:
        if re.search(r'( |^)%s(s|i|in|es)?( |$)' % (form), phrase):
            return True
    return False


def get_inout_grp(phrase, inoutlabels):
    def clean(s):
        symbols = r'(#|&|:|"|\?)'
        WHITESPACEREGEX = r'[ \t\n\r\f\v]+'
        s = re.sub(symbols, ' ', s)
        s = re.sub(WHITESPACEREGEX, ' ', s)
        return s.strip()

    phrase = clean(phrase)

    matches = []
    indexs = []
    occurs = []
    c = 0
    for form in inoutlabels:
        if re.search(r'( |^)%s(s|i|in|es)?( |$)' % (form), phrase):
            matches.append(form)
            indexs.append(re.search(r'( |^)%s(s|i|in|es)?( |$)' % (form), phrase).span(0)[0])
            occurs.append(c)
            c += 1

    # return the first match in sentence, if exist
    if len(matches) > 0:
        _, _, matches = zip(*sorted(zip(indexs, occurs, matches)))
        return matches[0]
    return None


def _is_aux_verb(toks):
    for tok in toks:
        if tok.pos_ == "AUX":
            return False
    return True


def postproverb(verb):
    filterl = ['be', 'am', "'m", 'is', 'are', "'re", 'has', 'have', 'had', 'by', 'in', 'is in', 'be in', 'will',
               'would', 'would be', 'will be', 'been', 'have been', 'had been', 'has been', 'were', 'was']

    if str(verb).lower().strip() in filterl:
        return False
    return True
