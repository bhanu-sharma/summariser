import os
import gc
import torch
import spacy
import pandas as pd
from tqdm import tqdm
from typing import List
from data_builder import BertData
from data_builder import greedy_selection

nlp = spacy.load("en")


def split_in_sentences(text: str):
    doc = nlp(text)
    para_sents = [[str(token) for token in sent] for sent in doc.sents]

    return para_sents


def _format_to_bert(input_doc_list: List[List], output_doc_list: List[List],
                    save_file: str, oracle_mode: str = "greedy"):
    if os.path.exists(save_file):
        print('Ignore %s' % save_file)
        return

    bert = BertData()

    datasets = []
    for source, tgt in tqdm(zip(input_doc_list, output_doc_list)):
        if oracle_mode == 'greedy':
            oracle_ids = greedy_selection(source, tgt, 3)
        # elif (args.oracle_mode == 'combination'):
        #     oracle_ids = combination_selection(source, tgt, 3)
        b_data = bert.preprocess(source, tgt, oracle_ids)
        if b_data is None:
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    print('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    gc.collect()


if __name__ == "__main__":
    df = pd.read_csv("./data/combined_training_data_130919.csv")
    df.cleaned_tool_input = df.cleaned_tool_input.astype("str")
    df.cleaned_output = df.cleaned_output.astype("str")

    tqdm.pandas()

    input_doc_list = df.cleaned_tool_input.progress_apply(split_in_sentences).to_list()
    output_doc_list = df.cleaned_output.progress_apply(split_in_sentences).to_list()

    _format_to_bert(input_doc_list, output_doc_list, "./data/bert_data.train.pt")

    # src = [sent.split() for sent in input_doc_list[0].split("\n")]
    # tgt = [output_doc_list[0].split()]

    # extracted_gold_data = []
    # for src, tgt in tqdm(zip(input_doc_list, output_doc_list)):
    #     selected_sentences = greedy_selection(src, tgt, 5)
    #     sentence_text = [src[i] for i in selected_sentences]
    #     extracted_gold_data.append(sentence_text)
