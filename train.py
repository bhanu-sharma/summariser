import os
import torch
import random
import argparse
from collections import namedtuple
from preprocess.data_loader import load_dataset

from model import model_builder
from preprocess import data_loader
from model.model_builder import Summarizer
from model.trainer import build_trainer

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(args, device_id, bert_data_path):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    print('Device ID %d' % device_id)
    print('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"out of something")

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(use_interval=args.use_interval,
                                      datasets=load_dataset(bert_data_path, 'train', shuffle=True),
                                      batch_size=args.batch_size, device=device,
                                      shuffle=True, is_test=False)

    print(f"loading model ...")
    model = Summarizer(args, device, load_pretrained_bert=True)
    print(f"model loaded")
    if args.train_from != '':
        print('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
        model.load_cp(checkpoint)
        optim = model_builder.build_optim(args, model, checkpoint)
    else:
        print(f"building optimiser")
        optim = model_builder.build_optim(args, model, None)

    print(model)
    print(f"building trainer")
    trainer = build_trainer(args, device_id, model, optim)
    print(f"trainer built, training commences")
    trainer.train(train_iter_fct, args.train_steps)


if __name__ == "__main__":
    args_dict = {'encoder': "classifier",
                 'mode': "train",
                 'bert_data_path': "../bert_data/cnndm",
                 'model_path': "./models/",
                 'result_path': "./results/",
                 'temp_dir': "./temp')",
                 'bert_config_path': "./bert_config_uncased_base.json",
                 'batch_size': 3000,
                 'use_interval': True,
                 'hidden_size': 128,
                 'ff_size': 512,
                 'heads': 4,
                 'inter_layers': 2,
                 'rnn_size': 512,
                 'param_init': 0,
                 'param_init_glorot': 'True',
                 'dropout': 0.1,
                 'optim': "adam",
                 'lr': 0.1,
                 'beta1': 0.9,
                 'beta2': 0.999,
                 'decay_method': "",
                 'warmup_steps': 500,
                 'max_grad_norm': 0,
                 'save_checkpoint_steps': 5,
                 'accum_count': 1,
                 'world_size': 1,
                 'report_every': 1,
                 'train_steps': 1000,
                 'recall_eval': 'False',
                 'visible_gpus': "-1",
                 'gpu_ranks': "0",
                 'log_file': "../logs/cnndm.log",
                 'dataset': "",
                 'seed': 666,
                 'test_all': 'False',
                 'test_from': "",
                 'train_from': "",
                 'report_rouge': 'False',
                 'block_trigram': 'True'}

    args = namedtuple("args", args_dict.keys())(**args_dict)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    train(args, device_id, bert_data_path="data/bert_data")
