import argparse
import gc
import logging
from typing import List, Dict, Optional

import flask
import spacy
import torch
from constants import EntityMap
from flask import jsonify, request
from models.model_builder import AbsSummarizer
from models.predictor import build_predictor
from others.logging import logger
from pytorch_transformers import BertTokenizer
from sd_preprocess.data_builder import BertData
from tqdm import tqdm

from models import data_loader

nlp = spacy.load("en")
entity_map = EntityMap()


def split_in_sentences(text: str):
    doc = nlp(text)
    para_sents = [[str(token) for token in sent] for sent in doc.sents]

    return para_sents


def _format_to_bert(
    input_doc_list: List[List[object]], output_doc_list: List[List[object]]
):

    bert = BertData()

    datasets = []
    for source, tgt in tqdm(zip(input_doc_list, output_doc_list)):
        oracle_ids = list(range(len(source)))

        b_data = bert.preprocess(source, tgt, oracle_ids)
        if b_data is None:
            continue
        src_subtoken_idx, sent_labels, tgt_subtoken_idx, segments_ids, cls_ids, src_txt, tgt_txt = (
            b_data
        )
        b_data_dict = {
            "src": src_subtoken_idx,
            "tgt": tgt_subtoken_idx,
            "src_sent_labels": sent_labels,
            "segs": segments_ids,
            "clss": cls_ids,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt,
        }
        datasets.append(b_data_dict)

    # print("Processed instances %d" % len(datasets))

    gc.collect()
    yield datasets


def test_abs(args, step, dataset):
    device = "cpu" if args.visible_gpus == "-1" else "cuda"
    test_from = args.test_from
    logger.info("Loading checkpoint from %s" % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    logging.info(args)

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(
        args, dataset, batch_size=50, device="cpu", shuffle=False, is_test=True
    )

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True, cache_dir=args.temp_dir
    )
    symbols = {
        "BOS": tokenizer.vocab["[unused0]"],
        "EOS": tokenizer.vocab["[unused1]"],
        "PAD": tokenizer.vocab["[PAD]"],
        "EOQ": tokenizer.vocab["[unused2]"],
    }
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predicted_string = predictor.translate_single(test_iter, step)
    return predicted_string


def _anonymise_text(tool_json: Dict[str, str]):
    tool_name = tool_json["tool_name"]
    tool_time = tool_json["tool_time"]
    candidate_name = tool_json["candidate_name"]

    job_duties = tool_json["job_duties"]

    anonymised_job_duties = (
        job_duties.replace(tool_name, entity_map.tool_name)
        .replace(tool_time, entity_map.time_spent)
        .replace(candidate_name.replace(",", "").title(), entity_map.name)
    )

    return anonymised_job_duties


def _deanonymise_text(tool_json: Dict[str, str], predicted_string: str):
    tool_name = tool_json["tool_name"]

    try:
        tool_time = tool_json["tool_time"].split("%")[0]
    except ValueError:
        tool_time = tool_json["tool_time"]

    try:
        first_name, last_name = tool_json["candidate_name"].split(",")
    except ValueError:
        last_name = tool_json["candidate_name"]

    deanonymised_job_duties = (
        predicted_string.replace(entity_map.tool_name, tool_name)
        .replace(entity_map.time_spent, tool_time)
        .replace(entity_map.name, last_name.title())
    )

    return deanonymised_job_duties


def create_model_input_text(tool_json: Dict[str, str], **kwargs):
    # A dummy function that will be hijacked when
    # some pre-processing would have to be done  on the text
    # received from the BE to the input text, like entity anonymisation.

    anonymised_job_duties = _anonymise_text(tool_json)

    input_text = (
        f"<NAME> {entity_map.name} <TOOL> {entity_map.tool_name} "
        f"<TIME> {entity_map.time_spent} <DATA> {anonymised_job_duties}"
    )
    return input_text


def update_and_jsonify_predicted_string(
    predicted_string: str, tool_json: Dict[str, str], **kwargs
):
    # A dummy function that will be hijacked when
    # some post-processing would have to be done on the predicted text
    # received from the model like filling the anonymised entities etc.
    deanonymised_string = _deanonymise_text(tool_json, predicted_string)

    try:
        tool_time = tool_json["tool_time"].split("%")[0]
    except ValueError:
        tool_time = tool_json["tool_time"]

    tool_name = tool_json["tool_name"].title()

    prediction_json = {
        "specialised_knowledge_paragraph": deanonymised_string,
        "tool_name": tool_name,
        "tool_time": tool_time,
    }

    return prediction_json


def generate_predictons(
    request: List[Dict[str, str]], arguments: argparse.Namespace
) -> List[Optional[Dict[str, str]]]:

    # TODO: Refactor this this man
    predictions = []

    # for tool_json in tqdm(request):
    input_text_list = [create_model_input_text(tool_json) for tool_json in request]
    dataset = _format_to_bert(
        [split_in_sentences(i) for i in input_text_list], [split_in_sentences(i) for i in input_text_list]
    )
    predicted_result = test_abs(arguments, dataset=dataset, step=0)

    for key, tool_json in tqdm(enumerate(request)):

        predicted_str = predicted_result[key]
        prediction_json = update_and_jsonify_predicted_string(
            predicted_string=predicted_str, tool_json=tool_json
        )

        predictions.append(prediction_json)

    return predictions


def get_flask_app(arguments: argparse.Namespace):
    app = flask.Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        predictions = []
        # ensure an image was properly uploaded to our endpoint
        if flask.request.method == "POST":
            data = request.get_json()
            if data:
                predictions = generate_predictons(request=data, arguments=arguments)

        #   return the prediction list as a JSON response
        return jsonify(predictions)

    return app


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-test_from",
        default="/Users/bhanu/OSS_Workspace/PreSumm/models/model_step_10000.pt",
    )
    parser.add_argument("-task", default="abs", type=str, choices=["ext", "abs"])
    parser.add_argument("-max_tgt_len", default=140, type=int)
    parser.add_argument("-max_pos", default=1000, type=int)
    parser.add_argument("-use_interval", default=True)
    parser.add_argument("-temp_dir", default="../temp")
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-recall_eval", default=False)
    parser.add_argument("-block_trigram", default=True)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument(
        "-encoder", default="bert", type=str, choices=["bert", "baseline"]
    )
    parser.add_argument("-model_path", default="../models/")
    parser.add_argument("-large", default=False)
    parser.add_argument("-share_emb", default=False)
    parser.add_argument("-finetune_bert", default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-port", default=5000, type=int)
    parser.add_argument("-visible_gpus", default="-1", type=str, help="set it 0 if want prediction from gpu else -1")
    # parser.add_argument('-gpu_ranks', default='0', type=str)

    args = parser.parse_args()
    app = get_flask_app(arguments=args)

    logger.info(
        (
            "* Loading Keras model and Flask starting server..."
            "please wait until server has fully started"
        )
    )
    app.run(host="0.0.0.0", debug=False, threaded=True, port=5000)
