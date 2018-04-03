import os

import re
import spacy


NLP = spacy.load('en')
LABELS = {
    'ENT': 'ENT',
    'PERSON': 'ENT',
    'NORP': 'ENT',
    'FAC': 'ENT',
    'ORG': 'ENT',
    'GPE': 'ENT',
    'LOC': 'ENT',
    'LAW': 'ENT',
    'PRODUCT': 'ENT',
    'EVENT': 'ENT',
    'WORK_OF_ART': 'ENT',
    'LANGUAGE': 'ENT',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': 'QUANTITY',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'CARDINAL'
}


def main():
    in_dir = "/Users/william/data/engineering_jd/"
    out_dir ="/Users/william/projects/sense2vec/data"

    all_docs = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    for i, doc_path in enumerate(all_docs):
        print(str(i), "out of", str(len(all_docs)), "docs processed")
        process_and_transform_doc(doc_path, out_dir)



def process_and_transform_doc(path, out_dir):
    with open(path, "r") as f:
        lines = [x.rstrip() for x in f.readlines()]

    out_filename= 'cleaned_'+ path.split(os.sep)[-1].split('.')[0] + ".txt"
    with open(os.path.join(out_dir, out_filename), "w") as o:
        for line in lines:
            if line.strip():
                sentence = NLP(line)
                prcd_text = process_sentence(sentence)
                o.write(prcd_text)


def process_sentence(doc):
    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])

    np_chunks = list(doc.noun_chunks)
    for np in np_chunks:
        np.merge(np.root.tag_, np.text, np.root.ent_type_)
    strings = []
    sents = list(doc.sents)

    for sent in sents:
        if sent.text.strip():
            strings.append(' '.join(represent_word(w) for w in sent if not w.is_space))
    if strings:
        return '\n'.join(strings) + '\n'
    else:
        return ''

def represent_word(word):
    if word.like_url:
        return '%%URL|X'
    text = re.sub(r'\s', '_', word.text)
    tag = LABELS.get(word.ent_type_, word.dep_)
    if not tag:
        tag = '?'
    return text + '|' + tag

if __name__=="__main__":
    main()