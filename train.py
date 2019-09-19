import os
import glob
import torch
import time
import random
import argparse
from collections import namedtuple
from data_loader import load_dataset

import data_loader, model_builder
from model_builder import Summarizer
from trainer import build_trainer
from pytorch_pretrained_bert import BertConfig

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def wait_and_validate(args, device_id):
    timestep = 0
    if args.test_all:
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args,  device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if i - max_step > 10:
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        print('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test(args,  device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args,  device_id, cp, step)
                    test(args,  device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args,  device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    print('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    valid_iter =data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test(args, device_id, pt, step):

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter,step)


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
            if (k in model_flags):
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
                 'bert_data_path': "./data/bert_data",
                 'model_path': "./models/",
                 'result_path': "./results/",
                 'temp_dir': "../temp')",
                 'bert_config_path': "../bert_config_uncased_base.json",
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
                 'test_all': False,
                 'test_from': "",
                 'train_from': "",
                 'report_rouge': 'False',
                 'block_trigram': 'True'}

    args = namedtuple("args", args_dict.keys())(**args_dict)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    # train(args, device_id, bert_data_path="data/bert_data")

    wait_and_validate(args, device_id)
