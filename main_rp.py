import os
import argparse

import ipdb
import torch
import wandb
import random
import json
import numpy as np
from transformers import T5Tokenizer, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoModelForCausalLM, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from zsre_dataset import Dataset, Sentence
from model import BaseT5
from encoding import select_encoder
from trainer import Trainer
from pathlib import Path
from model import ProtoT5
from models.T5withAdapter import T5ForConditionalGenerationWithAdapter


def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b

def compute_score_weight(path_pred, path_gold):
    pred = Dataset.load(path_pred)
    gold = Dataset.load(path_gold)

    num_pred = 0
    num_gold = 0
    num_correct = 0
    num_trip = 0

    for i in range(len(gold.sents)):
        num_pred += len(pred.sents[i].triplets)
        num_gold += len(gold.sents[i].triplets)
        for p in pred.sents[num_trip].triplets:
            for g in gold.sents[i].triplets:
                if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                    num_correct += 1
        num_trip += len(gold.sents[i].triplets)

    precision = safe_divide(num_correct, num_pred)
    recall = safe_divide(num_correct, num_gold)

    info = dict(
        # path_pred=path_pred,
        # path_gold=path_gold,
        precision=precision,
        recall=recall,
        score=safe_divide(2 * precision * recall, precision + recall),
    )
    return info

def compute_score(path_pred, path_gold):
    pred = Dataset.load(path_pred)
    gold = Dataset.load(path_gold)
    assert len(pred.sents) == len(gold.sents)
    num_pred = 0
    num_gold = 0
    num_correct = 0

    for i in range(len(gold.sents)):
        num_pred += len(pred.sents[i].triplets)
        num_gold += len(gold.sents[i].triplets)
        for p in pred.sents[i].triplets:
            for g in gold.sents[i].triplets:
                if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                    num_correct += 1

    precision = safe_divide(num_correct, num_pred)
    recall = safe_divide(num_correct, num_gold)

    info = dict(
        # path_pred=path_pred,
        # path_gold=path_gold,
        precision=precision,
        recall=recall,
        score=safe_divide(2 * precision * recall, precision + recall),
    )
    return info

def add_keyword(data,keyword,token_num=4):
    for sent in data.sents:
        key_token = keyword[sent.triplets[0].label_id]
        ss_temp = ' '.join(sent.triplets[0].tokens)
        temp_token, temp_token_num=[], 0
        for token in key_token:
            if ' '+token+' ' in ss_temp:
                temp_token.extend(token.split(' '))
                temp_token_num+=1
                if temp_token_num==token_num:
                    break
        if len(temp_token)!=0:
            key = list(set(temp_token))
            for trip in sent.triplets:
                trip.keyword = key
        else:
            for trip in sent.triplets:
                trip.keyword = ['None']


def init_args():
    parser = argparse.ArgumentParser("Zero-shot Relation Extraction")

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--do_train', default=0, type=int)
    parser.add_argument('--do_direct_eval', default=0, type=int)
    parser.add_argument('--do_eval', default=1, type=int)
    parser.add_argument('--keyword_path', default='./data/splits/zero_rte/fewrel/tfidf_kw.json', type=str)
    parser.add_argument('--t5_pretrain_model_path', default='../PretrainModel/t5-base', type=str)
    parser.add_argument('--gpt2_pretrain_model_path', default='../PretrainModel/gpt-2', type=str)
    # parser.add_argument('--gpt2_pretrain_model_path', default='./outputs/fewrel/unseen_5_seed_0/best_generator', type=str)
    parser.add_argument('--lr', default=3e-5, type=float, help='[3e-4, 1e-4]')
    parser.add_argument('--aux_lr', default=6e-4, type=float) #6e-4
    parser.add_argument('--aux_loss_weight', default=0.1, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--use_scheduler', default=1, type=int)
    parser.add_argument('--warmup_ratio', default=0, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--gradient_clip_val', default=1.0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--train_gen_epochs', default=3, type=int)
    parser.add_argument('--train_extract_epochs', default=3, type=int)
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--max_len', default=256, type=int)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_unseen', default=5, type=int)
    parser.add_argument('--task_num', default=1, type=int)
    parser.add_argument('--reptile_m', default=1, type=int)
    parser.add_argument('--dataset', default='fewrel', type=str, help='fewrel,wiki')
    args = parser.parse_args()

    args.pid2name_path = f'./data/splits/zero_rte/{args.dataset}/pid2name.json'
    args.train_path = f'./data/splits/zero_rte/{args.dataset}/unseen_{args.n_unseen}_seed_{args.seed}/train.jsonl'
    args.dev_path = f'./data/splits/zero_rte/{args.dataset}/unseen_{args.n_unseen}_seed_{args.seed}/dev.jsonl'
    args.test_path = f'./data/splits/zero_rte/{args.dataset}/unseen_{args.n_unseen}_seed_{args.seed}/test.jsonl'
    args.output_path = f'./outputs/{args.dataset}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.generate_path = f'./synthetic/{args.dataset}/unseen_{args.n_unseen}_seed_{args.seed}/synthetic.jsonl'
    return args

def process_new_data(data_sents,sents):
    for da,sent in zip(data_sents,sents):
        da.triplets[0].tokens += ['Keyword',':']+da.triplets[0].keyword

def train_generator(config):
    if not Path(config.generate_path).exists():
        if config.seed is not None:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)
            torch.backends.cudnn.deterministic = True
            print(f"Random seed is {config.seed}")

        encode_name = 'generate'  # extract
        encoder = select_encoder(encode_name)

        train_data = Dataset.load(config.train_path)
        dev_data = Dataset.load(config.dev_path)
        test_data = Dataset.load(config.test_path)
        # add_keyword(train_data,keyword=key_word)
        # add_keyword(dev_data,keyword=key_word)

        dev_labels = dev_data.get_labels()
        test_labels = test_data.get_labels()
        generate_labels = dev_labels + test_labels

        if Path(config.output_path + 'best_generator/').exists():
            gen_model = config.output_path + 'best_generator/'
        else:
            gen_model = None
        gpt_model = GPT2LMHeadModel.from_pretrained(config.gpt2_pretrain_model_path)
        gpt_tokenizer = GPT2Tokenizer.from_pretrained(config.gpt2_pretrain_model_path)

        Generator_Trainer = Trainer(config, gpt_model, gpt_tokenizer, encoder, train_data, train_data)
        Generator_Trainer.process_generation_data()
        Generator_Trainer.train_gen(model_path=gen_model)  #
        pipe = Generator_Trainer.make_pipe()
        groups = {}
        for relation in generate_labels:
            triplets, raw = Generator_Trainer.generate(relation, encoder, pipe)
            for t in triplets:
                groups.setdefault(t.text, []).append(t)
        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(config.generate_path)

def train_extractor(config):
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        print(f"Random seed is {config.seed}")

    encode_name = 'extract_rp'  # generate
    encoder = select_encoder(encode_name)

    t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrain_model_path)
    t5_tokenizer.add_tokens(['[PROTO]', '[HEAD]', '[TAIL]', '[REL]', '[SENT]'])

    # print(t5_tokenizer.encode('[PROTO]'))

    t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_pretrain_model_path)
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    model = ProtoT5(config, t5_tokenizer, t5_model, encoder)

    train_data = Dataset.load(config.train_path)
    dev_data = Dataset.load(config.dev_path)

    Extractor_Trainer = Trainer(config, model, t5_tokenizer, encoder, train_data, dev_data)
    Extractor_Trainer.process_extraction_data_rp()

    Extractor_path = Extractor_Trainer.train_extract_rp()
    # Extractor_path = config.output_path + 'best_extractor/'
    path_pred = str(config.output_path + 'pred.jsonl')

    Extractor_Trainer.predict_rp(config.test_path, path_pred, model_path=Extractor_path)
    results = compute_score(path_pred, config.test_path)
    # ipdb.set_trace()
    # Extractor_Trainer.predict_multi(config.test_path, path_pred, model_path=Extractor_path)

    print(json.dumps(results, indent=2))

    return config

def train_extractor_with_synthetic(config):
    train_data = Dataset.load(config.train_path)
    dev_data = Dataset.load(config.dev_path)
    extractor_path = config.output_path + 'best_extractor/'
    encode_name = 'extract_rp'  # generate
    encoder = select_encoder(encode_name)
    t5_tokenizer = T5Tokenizer.from_pretrained(extractor_path)
    t5_tokenizer.add_tokens(['[PROTO]', '[HEAD]', '[TAIL]', '[REL]', '[SENT]'])
    t5_model = T5ForConditionalGeneration.from_pretrained(extractor_path)

    t5_model.resize_token_embeddings(len(t5_tokenizer))
    model = ProtoT5(config, t5_tokenizer, t5_model, encoder)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        print(f"Random seed is {config.seed}")

    Extractor_Trainer = Trainer(config, model, t5_tokenizer, encoder, train_data, dev_data)
    synthetic_dataset = Dataset.load(config.generate_path)
    Extractor_Trainer.process_synthetic_data_rp(synthetic_dataset)

    Final_Extractor_Path = Extractor_Trainer.train_extract_rp(is_synthetic=True)
    # Final_Extractor_Path = config.output_path + 'best_extractor/'
    final_path_pred = str(config.output_path + 'final_pred.jsonl')
    Extractor_Trainer.predict(config.test_path, final_path_pred, model_path=Final_Extractor_Path)

    results = compute_score(final_path_pred, config.test_path)
    print(json.dumps(results, indent=2))
    with open(Path(config.output_path) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    # return results

if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "dryrun"  # 设置为离线
    opt = init_args()
    print(opt)
    # run = wandb.init(project='ZS-RTE',
    #                  job_type=opt.dataset,
    #                  name=f"{opt.dataset}-{opt.seed}")
    # wandb.config.update(opt)
    # train_generator(opt)
    # train_extractor(opt)
    train_extractor_with_synthetic(opt)
    # run.finish()