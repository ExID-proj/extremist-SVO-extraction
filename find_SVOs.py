
import re

import en_core_web_lg
from collections.abc import Iterable

from utils import passive_phrases, is_in_matches

# use spacy large model
nlp = en_core_web_lg.load()

# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}

# dependency tags for conjunctive verbs
CONJ = ["cc", "conj"]


# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet):
    return "and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs.extend(
                [tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN" or tok.pos_ == "PROPN"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))

    if len(more_objs) == 0:
        temp = objs.copy()
        for obj in temp:
            obj_children = [tok for tok in obj.children if tok.dep_ == 'conj']
            if len(obj_children) > 0:
                temp.extend(obj_children)
                more_objs.extend(obj_children)

    return more_objs


def _subjects(verb):
    subs = [tok for tok in verb.children if tok.dep_ in SUBJECTS]

    # Add subjects from conjunctions
    for sub in subs:
        sub_children = [tok for tok in sub.children if tok.dep_ == 'conj']
        if len(sub_children) > 0:
            subs.extend(sub_children)

    return subs


def _extract_subjects(verb, verbs):
    verb_negated = _is_negated(verb)

    parent_verb = None
    if verb.dep_ in CONJ:
        # If verb dependency is a conjunction, then find its "parent"
        for v in verbs:
            if verb in v.conjuncts:
                parent_verb = v
                break
        # Get parent's subject(s)
        if parent_verb is None:
            subs = _subjects(verb)
        else:
            subs = _subjects(parent_verb)
    else:
        subs = _subjects(verb)

    return subs, verb_negated


# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB" and str(tok).lower() not in ['who', 'whose']]
        if len(subs) == 0:
            subs = [tok for tok in head.lefts if tok.dep_ in SUBJECTS and str(tok).lower() not in ['who', 'whose']]

        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)

    elif head.pos_ == "NOUN":
        return [head], _is_negated(tok)

    return [], False


# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS and not re.findall('(not only)((\w+|\,)? ?){0,4}( ?but)?',
                                                      ' '.join([str(item) for item in parts])):
            return True
    return False


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_ in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# xcomp; open complement - verb has no subject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if
            tok.dep_ in SUBJECTS and tok.pos_ != "DET" and str(tok).lower() not in ['who', 'whose']]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated_ = _find_subs(v)
        subs.extend(foundSubs)

    return subs, verb_negated


# find the main verb - or any aux verb if we can't find it
def _find_verbs(tokens):
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    verbs_id = [idx for idx, tok in enumerate(tokens) if _is_non_aux_verb(tok)]
    if len(verbs) == 0:
        verbs = [tok for tok in tokens if _is_verb(tok)]
        verbs_id = [idx for idx, tok in enumerate(tokens) if _is_verb(tok)]
    return verbs, verbs_id


# is the token a verb?  (excluding auxiliary verbs)
def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass")


# is the token a verb?  (excluding auxiliary verbs)
def _is_verb(tok):
    return tok.pos_ == "VERB" or tok.pos_ == "AUX"


# return the verb to the right of this verb in a CCONJ relationship if applicable
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


def _extract_objects(verb):
    objs = [tok for tok in verb.children if tok.dep_ in OBJECTS]
    # Add object conjunctions
    for obj in objs:
        obj_children = [tok for tok in obj.children if tok.dep_ == 'conj']
        if len(obj_children) > 0:
            objs.extend(obj_children)

    # If verb has no objects, look for objects in "child" verb conjunctions
    if len(objs) == 0:
        verb_conj = [tok for tok in verb.children if tok.dep_ in CONJ and tok.pos_ == 'VERB']
        for v in verb_conj:
            v_objs = [tok for tok in v.children if tok.dep_ in OBJECTS]
            objs.extend(v_objs)

    return objs


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
        if v.tag_ == 'VB':
            is_pas = False

    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    else:
        objs.extend(_extract_objects(v))

    return v, is_pas, objs


# return true if there is a passive verb in sentence
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return None


# simple stemmer using lemmas
def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


# expand an obj / subj np using its chunk
def expand(item, tokens):
    if item.lower_ == 'that':
        temp_item = _get_that_resolution(tokens)
        if temp_item is not None:
            item = temp_item

    parts = []
    indexs = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)
                indexs.append(part.i)

    parts.append(item)
    indexs.append(item.i)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)
                indexs.append(part.i)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN" or item2.pos_ == "PROPN":
                parts.extend(expand(item2, tokens))
            break

    for i, tok in enumerate(parts):
        if tok.text == ',':
            parts.remove(tok)
            del indexs[i]
    if str(parts[-1]) == 'of':
        indchunks = {}
        for np in tokens.noun_chunks:
            indchunks[np.start] = np.text
        index = indexs[-1] + 1
        if index in indchunks:
            parts.append(nlp(indchunks[indexs[-1] + 1]))
            indexs.append(indexs[-1] + 1)

    return parts


# convert a list of tokens to a string
def to_str(tokens):
    if isinstance(tokens, Iterable):
        return ' '.join([item.text for item in tokens])
    else:
        return ''


def findSVOs(tokens):
    svos = []
    seenverbs = []

    verbs, verbs_id = _find_verbs(tokens)
    is_pas, _, m_indexes = passive_phrases(tokens)

    for v, vid in zip(verbs, verbs_id):
        if vid in seenverbs:
            continue
        seenverbs.append(vid)

        subs, verbNegated = _get_all_subs(v)

        if len(subs) == 0:
            subs, verbNegated = _extract_subjects(v, verbs)

        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            # multiple verbs
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, _, objs = _get_all_objs(conjV, is_in_matches(m_indexes, vid))
                seenverbs.append(v2.i)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas and is_in_matches(m_indexes, vid):  # reverse object / subject for passive
                            svos.append((to_str(expand(obj, tokens)),
                                         "!" + v.lemma_ if verbNegated or objNegated else v.lemma_,
                                         to_str(expand(sub, tokens))))
                            svos.append((to_str(expand(obj, tokens)),
                                         "!" + v2.lemma_ if verbNegated or objNegated else v2.lemma_,
                                         to_str(expand(sub, tokens))))
                        else:
                            svos.append((to_str(expand(sub, tokens)),
                                         "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                         to_str(expand(obj, tokens))))
                            svos.append((to_str(expand(sub, tokens)),
                                         "!" + v2.lower_ if verbNegated or objNegated else v2.lower_,
                                         to_str(expand(obj, tokens))))

            else:
                v, is_pas, objs = _get_all_objs(v, is_in_matches(m_indexes, vid))
                seenverbs.append(v.i)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas and is_in_matches(m_indexes, vid):  # reverse object / subject for passive
                            svos.append((to_str(expand(obj, tokens)),
                                         "!" + v.lemma_ if verbNegated or objNegated else v.lemma_,
                                         to_str(expand(sub, tokens))))
                        else:
                            svos.append((to_str(expand(sub, tokens)),
                                         "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                         to_str(expand(obj, tokens))))

    return svos
