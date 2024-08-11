import collections
import copy
import os
import pdb

import torch
import wandb
import random
from transformers import AdamW, get_linear_schedule_with_warmup, pipeline, Pipeline, set_seed
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from zsre_dataset import Dataset, Sentence
from transformers import T5Tokenizer, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoModelForCausalLM, T5ForConditionalGeneration
from models.T5withAdapter import T5ForConditionalGenerationWithAdapter
from encoding import ExtractEncoder
from encoding import select_encoder
from zsre_dataset import RelationSentence
from torch.nn import functional as F
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from utils import collate_fn, collate_fn_extract, collate_fn_extract_rc, find_sublist_index, find_sublist_start_end, collate_fn_extract_weight, collate_fn_extract_weight_predict, score_rc
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from pydantic import BaseModel
import shutil
import json
import ipdb

def delete_checkpoints(
    folder: str = ".", pattern="**/checkpoint*", delete: bool = True
):
    for p in Path(folder).glob(pattern):
        print(p)
        if delete:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                os.remove(p)
            else:
                raise ValueError("Unknown Type")

class LabelConstraint:
    def __init__(
        self,
        labels: List[str],
        tokenizer,
        prefix: str = " [REL] ",
    ):
        self.prefix: List[int] = tokenizer(prefix, add_special_tokens=False).input_ids
        self.label_map: Dict[int, str] = {
            tokenizer(" " + x, add_special_tokens=False).input_ids[0]: x for x in labels
        }
        self.tokenizer = tokenizer

    def run(self, triplet, scores):
        triplet = triplet.copy(deep=True)
        assert scores.ndim == 2
        token_ids = scores.argmax(dim=-1).int().tolist()
        i = find_sublist_index(token_ids, self.prefix)
        if i == -1:
            return triplet

        position = i + len(self.prefix)
        best = ""
        best_score = -1e9
        for j, label in self.label_map.items():
            score = scores[position, j].item()
            if score > best_score:
                best = label
                best_score = score

        if triplet.label in self.label_map.values():
            assert best == triplet.label

        assert len(best) > 0
        triplet.label = best
        triplet.score = best_score
        return triplet

class LabelConstraint_R1st:
    def __init__(
        self,
        labels: List[str],
        tokenizer,
        prefix: str = " Relation :",
    ):
        self.prefix: List[int] = tokenizer(prefix, add_special_tokens=False).input_ids
        self.label_map: Dict[int, str] = {
            tokenizer(" " + x, add_special_tokens=False).input_ids[0]: x for x in labels
        }
        self.tokenizer = tokenizer

    def run(self, triplet, scores):
        triplet = triplet.copy(deep=True)
        assert scores.ndim == 2
        token_ids = scores.argmax(dim=-1).int().tolist()
        i = find_sublist_index(token_ids, self.prefix)
        if i == -1:
            return triplet

        position = i + len(self.prefix)
        best = ""
        best_score = -1e9
        for j, label in self.label_map.items():
            score = scores[position, j].item()
            if score > best_score:
                best = label
                best_score = score

        if triplet.label in self.label_map.values():
            assert best == triplet.label

        assert len(best) > 0
        triplet.label = best
        triplet.score = best_score
        return triplet

class LabelConstraint_rp:
    def __init__(
        self,
        labels: List[str],
        tokenizer,
        prefix: str = ", Relation : ",
    ):
        self.prefix: List[int] = tokenizer(prefix, add_special_tokens=False).input_ids
        self.label_map: Dict[int, str] = {
            tokenizer(" " + x, add_special_tokens=False).input_ids[0]: x for x in labels
        }
        self.tokenizer = tokenizer

    def run(self, triplet, scores):
        triplet = triplet.copy(deep=True)
        assert scores.ndim == 2
        token_ids = scores.argmax(dim=-1).int().tolist()
        i = find_sublist_index(token_ids, self.prefix)
        if i == -1:
            return triplet

        position = i + len(self.prefix)
        best = ""
        best_score = -1e9
        for j, label in self.label_map.items():
            score = scores[position, j].item()
            if score > best_score:
                best = label
                best_score = score

        if triplet.label in self.label_map.values():
            assert best == triplet.label

        assert len(best) > 0
        triplet.label = best
        triplet.score = best_score
        return triplet

class DynamicModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

def get_proto_representation(pos_list, states, length=0):
    proto_reps = torch.stack([states[i,pos_list[i],:] for i in range(length)])
    return proto_reps

def get_marker_state(marker_position, decoder_hidden_state, marker_id=28898):
    B = marker_position.shape[0]
    marker_pos = [torch.where(marker_position[i]==marker_id)[-1][-1] for i in range(B)]
    marker_emb = torch.stack([decoder_hidden_state[i,pos,:] for i,pos in enumerate(marker_pos)])

    return marker_emb

class TextGenerator_weight(DynamicModel):
    model: T5ForConditionalGenerationWithAdapter
    tokenizer: T5Tokenizer
    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        batch,
        do_sample=True,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos = tok.eos_token_id#tok.bos_token_Id
        if 'decoder_input_ids' in batch:
            batch['decoder_input_ids'] = batch['decoder_input_ids'].to(self.model.device)
        if 'desc_pos' in batch:
            outputs = self.model.generate(
                **batch,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return,
                return_dict_in_generate=True,
                output_scores=save_scores,
                max_length=self.max_length,
            )
        else:
            outputs = self.model.generate(
                **batch,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return,
                return_dict_in_generate=True,
                output_scores=save_scores,
                max_length=self.max_length,
                **kwargs,
            )

        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        return self.decode(outputs.sequences)

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts

class TextGenerator(DynamicModel):
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        texts: List[str],
        do_sample=True,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        multi_prompt_ids: Optional[List[List[int]]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos = tok.eos_token_id#tok.bos_token_Id

        if prompt is not None:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if prompt_ids is not None:
            prompt_ids = [eos] + prompt_ids
            decoder_input_ids = torch.tensor([prompt_ids])
        if multi_prompt_ids is not None:
            assert len(texts) == len(multi_prompt_ids)
            multi_prompt_ids = [[eos] + lst for lst in multi_prompt_ids]
            decoder_input_ids = torch.tensor(multi_prompt_ids)
        if decoder_input_ids is not None:
            kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))
            kwargs.update(decoder_attention_mask=decoder_attention_mask.to(self.model.device))

        outputs = self.model.generate(
            **self.tokenize(texts),
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return,
            return_dict_in_generate=True,
            output_scores=save_scores,
            max_length=self.max_length,
            **kwargs,
        )

        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        return self.decode(outputs.sequences)

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts

class TripletSearchDecoder(DynamicModel):
    gen: TextGenerator
    constraint: LabelConstraint
    encoder: ExtractEncoder
    top_k: int = 4
    prompt: str = ''
    def generate(self, text: str, **kwargs) -> Tuple[str, Tensor]:
        outputs = self.gen.run(
            [text],
            do_sample=False,
            num_return=1,
            num_beams=1,
            save_scores=True,
            **kwargs,
        )

        assert len(outputs) == 1
        assert self.gen.scores is not None
        scores = torch.log_softmax(self.gen.scores[0], dim=-1)
        assert scores.ndim == 2
        return outputs[0], scores

    def find_prefix_end(self, token_ids: List[str], prefix: str) -> int:
        prefix_ids = self.gen.tokenizer(prefix, add_special_tokens=False).input_ids
        i = find_sublist_index(token_ids, prefix_ids)
        position = i + len(prefix_ids)
        return position

    def branch(
        self, text: str, prefix: str, prompt: Optional[str] = None, **kwargs
    ) -> List[Tuple[str, float]]:

        _, scores = self.generate(text, prompt=prompt, **kwargs)
        token_ids = scores.argmax(dim=-1).int().tolist()
        i = self.find_prefix_end(token_ids, prefix)

        pairs = []
        for j in torch.argsort(scores[i])[-self.top_k :]:
            p = (prompt or "") + self.gen.decode([token_ids[:i] + [j]])[0]
            pairs.append((p, scores[i, j].item()))

        return pairs

    def run(self, text: str) -> List[RelationSentence]:
        x = self.prompt + self.encoder.encode_x(text)
        outputs = []

        for prompt_a, score_a in self.branch(x, prefix="[HEAD]"): #Head Entity :
            for prompt_b, score_b in self.branch(
                x, prefix="[TAIL]", prompt=prompt_a    # Tail Entity :
            ):
                output, scores = self.generate(x, prompt=prompt_b)
                token_ids = token_ids = scores.argmax(dim=-1).int().tolist()
                i = self.find_prefix_end(token_ids, prefix="[REL]")    #Relation :
                score_c = max(scores[i].tolist())
                s = self.encoder.safe_decode(x=x, y=output)
                s = self.constraint.run(s, scores)
                # score_c = s.score  # From LabelConstraint
                s.score = (score_a + score_b + score_c) / 3
                outputs.append(s)

        return outputs

class Trainer:
    def __init__(self, opt, model, tokenizer, encoder, train_data, dev_data, d_model=768, proto_method=True):
        self.opt = opt
        self.model = model
        self.proto_method = proto_method
        # self.neg_cls = nn.Linear(d_model,2)
        self.tokenizer = tokenizer
        self.encoder = encoder
        if torch.cuda.is_available():
            self.model.cuda()

        self.train_dataset, self.dev_dataset = train_data, dev_data
        self.random_seed = 42
        self.pipe_name='text-generation'
        self.train_labels = train_data.get_labels()
        self.dev_labels = dev_data.get_labels()
        self.pid2name={}

        self.rel_prefix = tokenizer('Relation', add_special_tokens=False).input_ids
        self.relation_linear = nn.Linear(d_model, d_model, bias=False)
        self.relation_loss = nn.CrossEntropyLoss()

    def process_generation_data(self):
        self.opt.epochs = self.opt.train_gen_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        for sent in self.train_dataset.sents:
            train_sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
            temp_train = [self.tokenizer(sent, return_tensors='pt') for sent in train_sents]
            self.train_data.extend(temp_train)

        for sent in self.dev_dataset.sents:
            dev_sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
            temp_dev = [self.tokenizer(sent, return_tensors='pt') for sent in dev_sents]
            self.dev_data.extend(temp_dev)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)
        
        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=4,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn)

    def get_prompt(self, train_labels, t_label, task_num=3):#train_labels:*4那部分   t_label：每一轮拿出来的那一个
        # pdb.set_trace()
        prompt_list = []
        temp = copy.deepcopy(train_labels)
        assert t_label in temp
        temp.remove(t_label)
        prompt_text_list = []
        int_num = []
        for i in range(task_num):
            temp_prompt = []
            # temp_labels = random.sample(temp,k=self.opt.n_unseen-1)
            temp_labels = random.sample(temp,k=1)
            # rand_int = random.randint(0,self.opt.n_unseen-1)
            rand_int = random.randint(0,2)
            temp_prompt.extend(temp_labels[:rand_int])
            temp_prompt.append(t_label)
            temp_prompt.extend(temp_labels[rand_int:])
            prompt_list.append('[PROTO] ' + ', '.join(temp_prompt)+'. ')
            # prompt_list.append('[PROTO] ' + t_label +'. ')
            prompt_text_list.append(temp_prompt)
            int_num.append(rand_int)
        return prompt_list, prompt_text_list, int_num

    def get_prompt_multi(self, train_labels, t_label_list, task_num=3):
        prompt_list = []
        temp = copy.deepcopy(train_labels)
        t_label_list = set(t_label_list)
        for t_label in t_label_list:
                temp.remove(t_label)

        prompt_text_list = []
        int_num = []
        for i in range(task_num):
            temp_prompt = []
            try:
                temp_labels = random.sample(temp,k=self.opt.n_unseen-len(t_label_list))
            except:
                temp_labels = []
            temp_prompt.extend(temp_labels)
            temp_prompt.extend(t_label_list)
            random.shuffle(temp_prompt)
            prompt_text_list.append(temp_prompt)
            prompt_list.append('[PROTO] ' + ', '.join(temp_prompt) + '. ')

        return prompt_list, prompt_text_list, int_num

    def get_neg_prompt(self,train_labels, t_label, rel_num=0):
        prompt = ['[PROTO]']
        temp = copy.deepcopy(train_labels)
        temp.remove(t_label)
        assert t_label not in temp
        temp_labels = random.choices(temp,k=rel_num)
        prompt.extend(temp_labels)
        return ', '.join(prompt)+'. '

    def get_tex_len(self, text):
        return self.tokenizer(text, return_tensors='pt',
                       add_special_tokens=False)['input_ids'].shape[1]

    def process_extraction_data_1(self, use_prompt=True):
        generate_encoder = select_encoder('generate')

        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        for sent in self.train_dataset.sents:
            train_generate_sents = [generate_encoder.encode_to_line(trip) for trip in sent.triplets]
            generate_train = [self.tokenizer(sent, return_tensors='pt') for sent in train_generate_sents]
            generate_input = [self.tokenizer('Relation :' + trip.label+' .', return_tensors='pt') for trip in sent.triplets]
            train_sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
            trip_labels = [trip.label for trip in sent.triplets]

            for sent, t_label, generate, gen_inp in zip(train_sents,trip_labels, generate_train, generate_input):
                temp_train = {}
                context, labels = self.encoder.parse_line(sent)
                if use_prompt:
                    prompt_list, labels_text, proto_label = self.get_prompt(self.train_labels, t_label)
                    context = prompt_list[0] + context
                text = self.tokenizer(context, return_tensors='pt')
                prompt_labels = self.tokenizer(labels, return_tensors='pt')
                temp_train['input_ids'] = text['input_ids']
                temp_train['attention_mask'] = text['attention_mask']
                temp_train['decoder_input_ids'] = prompt_labels['input_ids']
                temp_train['decoder_attention_mask'] = prompt_labels['attention_mask']
                temp_train['labels'] = t_label
                temp_train['generate_input_ids'] = gen_inp['input_ids']
                temp_train['generate_attention_mask'] = gen_inp['attention_mask']
                temp_train['generate_decoder_labels'] = generate['input_ids']

                self.train_data.append(temp_train)

        for sent in self.dev_dataset.sents:
            dev_sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
            trip_labels = [trip.label for trip in sent.triplets]
            for sent, d_label in zip(dev_sents, trip_labels):
                temp_dev = {}
                context, labels = self.encoder.parse_line(sent)
                if use_prompt:
                    prompt_list, labels_text, proto_label = self.get_prompt(self.dev_labels, d_label)
                    context = prompt_list[0] + context
                text = self.tokenizer(context, return_tensors='pt')
                prompt_labels = self.tokenizer(labels, return_tensors='pt')
                temp_dev['input_ids'] = text['input_ids']
                temp_dev['attention_mask'] = text['attention_mask']
                temp_dev['decoder_input_ids'] = prompt_labels['input_ids']
                temp_dev['decoder_attention_mask'] = prompt_labels['attention_mask']
                # temp_dev['labels'] = d_label
                self.dev_data.append(temp_dev)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=1,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_extract)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_extract)

    #核心：把自己加的东西(除关系类型用字符串）转换成  '键'：[tensor/数字/数组]  的形式     data：[{}]
    def process_extrat_metric(self, dataset, labels_set, use_prompt=True):
        #add prototype
        data = []
        if len(labels_set)<self.opt.n_unseen:
            labels_set = labels_set*4
        for sent in dataset.sents:#dataset.sents中有其他位置信息(是所有的数据集信息)是一个含sentence对象的列表
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]#句子+三元组-->列表(为了一会分开处理句子和三元组)
            trips = [trip for trip in sent.triplets]#所有的数据集信息-->列表(为了一会提取位置信息)

            for sent, trip in zip(sents,trips):#从列表中拿出来
                temp_train = {}
                context, labels = self.encoder.parse_line(sent)#句子，三元组(labels)分开，分别构建encoder端和decoder端
                # encoder端：
                if use_prompt:
                    prompt_list, labels_text, proto_label_int = self.get_prompt(labels_set, trip.label, self.opt.task_num)#trip.label:关系类型，labels：三元组
                    # prompt_list:['[PROTO] religion, religion. ']    labels_text:[['religion', 'religion']]    proto_label_int:[1]
                    context_list = [prompt + context for prompt in prompt_list]#原型+关系提示+句子
                    #['[PROTO] religion, religion. [SENT] : Johannes Joseph ... Aachen .']
                else:
                    proto_label = None

                text={}
                text['input_ids']=[]
                text['attention_mask']=[]
                for context in context_list:
                    temp = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)#原型+关系提示+句子(encoder端)的序列化表示
                    text['input_ids'].append(temp['input_ids'])
                    text['attention_mask'].append(temp['attention_mask'])

                #decoder端：
                prompt_labels = self.tokenizer(labels, return_tensors='pt')#将三元组标签序列化
                #labels='[HEAD] Johannes Joseph van der Velden , [TAIL] Catholic , [REL] religion .'
                # 'Context : ': 1193,  6327,     3,    10
                # context_pos = torch.where(text['input_ids'] == 1193)[1].tolist()

                #加入到总代码
                temp_train['input_ids'] = text['input_ids']
                temp_train['attention_mask'] = text['attention_mask']
                temp_train['decoder_input_ids'] = [prompt_labels['input_ids'] for i in range(len(context_list))]#三元组标签的序列化表示加入
                temp_train['decoder_attention_mask'] = [prompt_labels['attention_mask'] for i in range(len(context_list))]
                temp_train['labels'] = [trip.label for i in range(len(context_list))]
                # ipdb.set_trace()
                # HEAD, TAIL

                # encoder端：
                try:
                    head_tokens_upper = ' '.join(trip.tokens[:trip.head[0]])
                    head_tokens = ' '.join(trip.tokens[trip.head[0]:trip.head[-1]+1])
                    tail_tokens_upper = ' '.join(trip.tokens[:trip.tail[0]])
                    tail_tokens = ' '.join(trip.tokens[trip.tail[0]:trip.tail[-1]+1])
                    # assert head_tokens in context
                    # assert tail_tokens in context
                except:
                    break
                head_pos, tail_pos =[], []
                head_len1 = self.get_tex_len(head_tokens_upper)#头实体序列化之后的开始位置
                head_len2 = head_len1 + self.get_tex_len(head_tokens)#头实体序列化之后的结束位置
                tail_len1 = self.get_tex_len(tail_tokens_upper)
                tail_len2 = tail_len1 + self.get_tex_len(tail_tokens)

                for prom in prompt_list:
                    temp_head, temp_tail=[], []
                    prom_len = self.get_tex_len(prom)#原型长度
                    temp_head.append(prom_len+head_len1)#加上原型之后的头实体序列化开始位置
                    temp_head.append(prom_len+head_len2)#加上原型之后的头实体序列化结束位置
                    temp_tail.append(prom_len+tail_len1)
                    temp_tail.append(prom_len+tail_len2)

                    head_pos.append(temp_head)
                    tail_pos.append(temp_tail)
                temp_train['head_pos'] = head_pos#什么时候需要位置：只有当有句子、没单独把实体拿出来的时候需要
                temp_train['tail_pos'] = tail_pos

                #decoder端：
                #找到序列化之后的三元组(标签：[HEAD]、[TAIL]、[REL])原型位置：
                temp_train['head_dec_proto'] = torch.where(prompt_labels['input_ids'] == 32101, )[1].tolist()   # [HEAD]  32101
                temp_train['tail_dec_proto'] = torch.where(prompt_labels['input_ids'] == 32102, )[1].tolist()   # [TAIL]  32102
                temp_train['rel_dec_proto'] = torch.where(prompt_labels['input_ids'] == 32103, )[1].tolist()   # [REL]  32102

                # REL
                rel_pos = []
                rel_idx = self.tokenizer(trip.label, return_tensors='pt', add_special_tokens=False)['input_ids']
                rel_len = rel_idx.shape[1]  #rel_idx:tensor([[5562]])

                #encoder端：
                for i,label_text in enumerate(labels_text):#labels_text：[['religion', 'religion']]
                    # 找到序列化之后关系(值，非标签)的位置(利用序列化之后的起始位置字的tensor值是否行相等来判断)
                    #['[PROTO] religion, religion. [SENT] : Johannes Joseph van ... Aachen .']中的religion位置
                    start_list = torch.where(text['input_ids'][i] == rel_idx[0,0],)[1].tolist()
                    for start in start_list:
                        end = start + rel_len
                        #用整个词的序列化结果来验证
                        if text['input_ids'][i][0][start:end].equal(rel_idx[0]):
                            rel_pos.append([start,end])
                            break
                assert len(rel_pos) == len(prompt_list)

                temp_train['rel_pos'] = rel_pos #加入关系类型的位置
                # proto_label_list = [0]*self.opt.n_unseen
                # proto_label_list[proto_label_int] = 1
                # temp_train['rel_label_int'] = proto_label_list
                # temp_train['rel_dec_proto'] = torch.where(prompt_labels['input_ids'] == 32103, )[1].tolist()   # [REL]  32103


                # if with_decode:
                #     prompt_context, labels_text, proto_label = self.get_prompt(labels_set, t_label, rel_num=1)
                #     context_new = prompt_context+context_neg
                #     text = self.tokenizer(context_new, return_tensors='pt', add_special_tokens=True)
                #     pos = torch.where(text['input_ids']==1193)[1].tolist()
                #     temp_train['context_pos'] = [pos[0],pos[0]+1]
                #     temp_train['input_ids_con'] = text['input_ids']
                #     temp_train['attention_mask_con'] = text['attention_mask']
                #     prompt_neg = self.get_neg_prompt(labels_set, t_label)
                #     context_neg = prompt_neg+context_neg
                #     text_neg = self.tokenizer(context_neg, return_tensors='pt', add_special_tokens=True)
                #     temp_train['input_ids_neg'] = text_neg['input_ids']
                #     temp_train['attention_mask_neg'] = text_neg['attention_mask']
                #     pos = torch.where(text_neg['input_ids'] == 1193)[1].tolist()
                #     temp_train['context_pos_neg'] = [pos[0], pos[0] + 1]

                data.append(temp_train)
        return data

    def process_extrat_rp(self, dataset, labels_set):
        #add prototype
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
            trips = [trip for trip in sent.triplets]

            for sent, trip in zip(sents,trips):
                temp_train = {}
                context, labels = self.encoder.parse_line(sent)

                text={}
                text['input_ids']=[]
                text['attention_mask']=[]
                temp = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                text['input_ids'].append(temp['input_ids'])
                text['attention_mask'].append(temp['attention_mask'])
                prompt_labels = self.tokenizer(labels, return_tensors='pt')
                # 'Context : ': 1193,  6327,     3,    10
                # context_pos = torch.where(text['input_ids'] == 1193)[1].tolist()

                temp_train['input_ids'] = text['input_ids']
                temp_train['attention_mask'] = text['attention_mask']
                temp_train['decoder_input_ids'] = [prompt_labels['input_ids']]
                temp_train['decoder_attention_mask'] = [prompt_labels['attention_mask']]
                temp_train['labels'] = [trip.label]
                data.append(temp_train)
        return data

    def process_extrat_reptile_multi(self, dataset, labels_set, use_prompt=True):
        #add prototype
        data = []
        if len(labels_set)<self.opt.n_unseen:
            labels_set = labels_set*4
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
            trips = [trip for trip in sent.triplets]

            trips_label = [trip.label for trip in sent.triplets]

            prompt_list, labels_text, proto_label_int = self.get_prompt_multi(labels_set, trips_label, self.opt.task_num)
            labels=''
            for sent in sents:
                context, temp_labels = self.encoder.parse_line(sent)
                labels += temp_labels+' '

            context_list = [prompt + context for prompt in prompt_list]

            text, temp_train={},{}
            text['input_ids']=[]
            text['attention_mask']=[]
            for context in context_list:
                temp = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                text['input_ids'].append(temp['input_ids'])
                text['attention_mask'].append(temp['attention_mask'])
            prompt_labels = self.tokenizer(labels.strip(), return_tensors='pt')
            # 'Context : ': 1193,  6327,     3,    10
            # context_pos = torch.where(text['input_ids'] == 1193)[1].tolist()

            temp_train['input_ids'] = text['input_ids']
            temp_train['attention_mask'] = text['attention_mask']
            temp_train['decoder_input_ids'] = [prompt_labels['input_ids'] for i in range(len(context_list))]
            temp_train['decoder_attention_mask'] = [prompt_labels['attention_mask'] for i in range(len(context_list))]
            data.append(temp_train)
        return data

    def process_extrat_reptile(self, dataset, labels_set, use_prompt=True, is_rc=False):
        #add prototype
        data = []
        if len(labels_set)<self.opt.n_unseen:
            labels_set = labels_set*4
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
            trips = [trip for trip in sent.triplets]

            for sent, trip in zip(sents,trips):
                temp_train = {}
                context, labels = self.encoder.parse_line(sent)
                if use_prompt:
                    if is_rc:
                        ht_en, _ = labels.split(', [REL]')
                        # ht_en_ids = self.tokenizer(ht_en, return_tensors='pt', add_special_tokens=False)['input_ids']
                        prompt_list, labels_text, proto_label_int = self.get_prompt(labels_set, trip.label,
                                                                                    self.opt.task_num)
                        context_list = [prompt + ht_en  + ' . ' + context for prompt in prompt_list]
                    else:
                        prompt_list, labels_text, proto_label_int = self.get_prompt(labels_set, trip.label, self.opt.task_num)
                        context_list = [prompt + context for prompt in prompt_list]
                else:
                    proto_label = None
                text={}
                text['input_ids']=[]
                text['attention_mask']=[]
                for context in context_list:
                    temp = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                    text['input_ids'].append(temp['input_ids'])
                    text['attention_mask'].append(temp['attention_mask'])
                prompt_labels = self.tokenizer(labels, return_tensors='pt')
                # 'Context : ': 1193,  6327,     3,    10
                # context_pos = torch.where(text['input_ids'] == 1193)[1].tolist()

                temp_train['input_ids'] = text['input_ids']
                temp_train['attention_mask'] = text['attention_mask']
                temp_train['decoder_input_ids'] = [prompt_labels['input_ids'] for i in range(len(context_list))]
                temp_train['decoder_attention_mask'] = [prompt_labels['attention_mask'] for i in range(len(context_list))]
                temp_train['labels'] = [trip.label for i in range(len(context_list))]
                data.append(temp_train)
        return data

    def process_extrat_weight(self, dataset, labels_set):
        
        rel_dict = {rel:[] for rel in labels_set} #{'nominated for': [], 'religion': []}
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_train = {}
                context, labels = self.encoder.parse_line(sent)# content:sent["text"]   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                temp_train['input_ids'], temp_train['attention_mask'] = [], []

                temp = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp['input_ids'], temp['attention_mask']

                temp = self.tokenizer(labels, return_tensors='pt')
                temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp['input_ids'], temp['attention_mask']
                temp_train['labels'] = trip.label # trip.label = 'religion'
                rel_dict[trip.label].append(temp_train)

        ipdb.set_trace()
        for true_rel, sents in rel_dict.items():
            for sent in sents:
                temp_prompt,temp_data = [],{}
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                # rel_idx = [rel_idx_dict[rel] for rel in candidate_rel]
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(true_rel) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])
                prompt = '[PROTO1] [PROTO2] [PROTO3]' + ', '.join(temp_prompt)+ '. ' #'[PROTO1] [PROTO2] [PROTO3]participant in, position held, constellation, member of, after a work by. '
                temp = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp['input_ids'], temp['attention_mask']

                candidate_sents = [random.choice(rel_dict[rel]) for rel in candidate_rel] #四种候选集的例子采样
                temp_data['sent'] = sent # 是一个字典，包含：正确关系的句子对应的input_ids、attention_mask、decoder_input_ids、decoder_attention_mask、labels
                temp_data['candidate_sents'] = candidate_sents #是一个列表。包含其他采样关系(非真正)的sent字典
                temp_data['prompt_ids'] = prompt_ids
                temp_data['prompt_attn'] = prompt_attn
                data.append(temp_data)
        return data

    def process_extrat_weight_predict(self, dataset, labels_set, prompt=''):
        data = []
        # No relation
        if len(labels_set)<self.opt.n_unseen:
            labels_set = labels_set*4
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
            trips = [trip for trip in sent.triplets]
            for sent, trip in zip(sents, trips):
                temp_train = {}
                context, labels = self.encoder.parse_line(sent)

                temp = self.tokenizer(prompt + context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'] = temp['input_ids']
                temp_train['attention_mask'] = temp['attention_mask']

                prompt_labels = self.tokenizer(labels, return_tensors='pt')

                temp_train['decoder_input_ids'] = prompt_labels['input_ids']
                temp_train['decoder_attention_mask'] = prompt_labels['attention_mask']
                data.append(temp_train)
        return data

    # def process_extrat_weight(self, dataset, labels_set, use_desc=True, use_prompt=True, prompt_num=1, prompt=None):
    #     data = []
    #     # No relation
    #     if len(labels_set)<self.opt.n_unseen:
    #         labels_set = labels_set*4
    #     for sent in dataset.sents:
    #         sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
    #         trips = [trip for trip in sent.triplets]
    #         for sent, trip in zip(sents, trips):
    #             temp_train = {}
    #             context, labels = self.encoder.parse_line(sent)
    #             if use_prompt:
    #                 if prompt is None:
    #                     prompt_list, labels_text, proto_label_int = self.get_prompt(labels_set, trip.label, prompt_num)
    #                     context_list = [prompt + context for prompt in prompt_list]
    #             else:
    #                 context_list = context
    #
    #             temp_train['input_ids'], temp_train['attention_mask'] = [], []
    #             for context in context_list:
    #                 temp = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
    #                 temp_train['input_ids'].append(temp['input_ids'])
    #                 temp_train['attention_mask'].append(temp['attention_mask'])
    #                 #self.tokenizer('[sent]', return_tensors='pt', add_special_tokens=True)
    #                 # prompt_len.append(torch.where(temp['input_ids'] == 32104)[1].tolist()[0])
    #
    #             prompt_labels = self.tokenizer(labels, return_tensors='pt')
    #
    #             temp_train['decoder_input_ids'] = [prompt_labels['input_ids'] for i in range(self.opt.task_num)]
    #             temp_train['decoder_attention_mask'] = [prompt_labels['attention_mask'] for i in
    #                                                     range(len(context_list))]
    #             temp_train['labels'] = [trip.label for i in range(len(context_list))]
    #             temp_train['prompt_input_ids'], temp_train['prompt_attention_mask'] = [],[]
    #             if use_desc:
    #                 desc_list, temp_train['desc_pos'] = [], []
    #                 for labels in labels_text:
    #                     for l in labels:
    #                         assert l in self.pid2name
    #                         desc_list.append(l + ' [DESC] ' + self.pid2name[l])
    #                 temp_list = [self.tokenizer(desc, return_tensors='pt', add_special_tokens=True) for desc in desc_list]
    #                 for temp in temp_list:
    #                     temp_train['prompt_input_ids'].append(temp['input_ids'])
    #                     temp_train['prompt_attention_mask'].append(temp['attention_mask'])
    #                     temp_train['desc_pos'].append(torch.where(temp['input_ids'] == 32105)[1].tolist()[0])
    #             data.append(temp_train)
    #     return data

    # def process_extrat_weight(self, dataset, labels_set, use_desc=True, use_prompt=True, prompt_num=1, prompt=None):
    #     data = []
    #     # No relation
    #     if len(labels_set)<self.opt.n_unseen:
    #         labels_set = labels_set*4
    #     for sent in dataset.sents:
    #         sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]
    #         trips = [trip for trip in sent.triplets]
    #         for sent, trip in zip(sents, trips):
    #             temp_train = {}
    #             context, labels = self.encoder.parse_line(sent)
    #             if use_prompt:
    #                 if prompt is None:
    #                     prompt_list, labels_text, proto_label_int = self.get_prompt(labels_set, trip.label, prompt_num)
    #                     context_list = [prompt + context for prompt in prompt_list]
    #             else:
    #                 context_list = context
    #
    #             temp_train['input_ids'], temp_train['attention_mask'] = [], []
    #             for context in context_list:
    #                 temp = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
    #                 temp_train['input_ids'].append(temp['input_ids'])
    #                 temp_train['attention_mask'].append(temp['attention_mask'])
    #                 #self.tokenizer('[sent]', return_tensors='pt', add_special_tokens=True)
    #                 # prompt_len.append(torch.where(temp['input_ids'] == 32104)[1].tolist()[0])
    #
    #             prompt_labels = self.tokenizer(labels, return_tensors='pt')
    #
    #             temp_train['decoder_input_ids'] = [prompt_labels['input_ids'] for i in range(self.opt.task_num)]
    #             temp_train['decoder_attention_mask'] = [prompt_labels['attention_mask'] for i in
    #                                                     range(len(context_list))]
    #             temp_train['labels'] = [trip.label for i in range(len(context_list))]
    #             temp_train['prompt_input_ids'], temp_train['prompt_attention_mask'] = [],[]
    #             if use_desc:
    #                 desc_list, temp_train['desc_pos'] = [], []
    #                 for labels in labels_text:
    #                     for l in labels:
    #                         assert l in self.pid2name
    #                         desc_list.append(l + ' [DESC] ' + self.pid2name[l])
    #                 temp_list = [self.tokenizer(desc, return_tensors='pt', add_special_tokens=True) for desc in desc_list]
    #                 for temp in temp_list:
    #                     temp_train['prompt_input_ids'].append(temp['input_ids'])
    #                     temp_train['prompt_attention_mask'].append(temp['attention_mask'])
    #                     temp_train['desc_pos'].append(torch.where(temp['input_ids'] == 32105)[1].tolist()[0])
    #             data.append(temp_train)
    #     return data

    def process_extraction_data_weight(self, pid2name=''):
        
        with open(pid2name) as f:
            pid2name = json.load(f)
            for key,values in pid2name.items():
                self.pid2name[values[0]]=values[1]
        #Oxford english dictionary
        # wrong_lb = []
        # for lb in self.train_labels:
        #     if lb not in self.pid2name:
        #         wrong_lb.append(lb)
        # for lb in self.dev_labels:
        #     if lb not in self.pid2name:
        #         wrong_lb.append(lb)
        # print(wrong_lb)

        self.pid2name['participant in'] = 'an active participant in the negotiations'
        self.pid2name['distributed by'] = 'something is distributed by somebody'
        self.pid2name['original broadcaster'] = 'an original company that sends out television or radio programmes'

        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_extrat_weight(self.train_dataset,self.train_labels)
        self.dev_data = self.process_extrat_weight_predict(self.dev_dataset,self.dev_labels)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        # ipdb.set_trace()
        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=2,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_extract_weight)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_extract_weight_predict)

    def process_extraction_data_metric(self, use_prompt=True, with_decode=False):
        #only extract
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_extrat_metric(self.train_dataset,self.train_labels,use_prompt)
        self.dev_data = self.process_extrat_metric(self.dev_dataset,self.dev_labels,use_prompt)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=1,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_extract)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_extract)

    def process_extraction_data_rp(self):
        #only extract
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_extrat_rp(self.train_dataset,self.train_labels)
        self.dev_data = self.process_extrat_rp(self.dev_dataset,self.dev_labels)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=2,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_extract)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_extract)

    def process_extraction_data_reptile_multi(self, use_prompt=True, with_decode=False):
        #only extract
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_extrat_reptile_multi(self.train_dataset,self.train_labels,use_prompt)
        self.dev_data = self.process_extrat_reptile_multi(self.dev_dataset,self.dev_labels,use_prompt)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=2,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_extract)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_extract)

    def process_extraction_data_reptile(self, use_prompt=True):
        #only extract
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_extrat_reptile(self.train_dataset,self.train_labels,use_prompt)
        self.dev_data = self.process_extrat_reptile(self.dev_dataset,self.dev_labels,use_prompt)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=2,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_extract)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_extract)

    def process_extraction_data_rc(self, use_prompt=True):
        #only extract
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_extrat_reptile(self.train_dataset,self.train_labels,use_prompt,is_rc=True)
        self.dev_data = self.process_extrat_reptile(self.dev_dataset,self.dev_labels,use_prompt,is_rc=True)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=2,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_extract_rc)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_extract_rc)

    def process_synthetic_data(self, synthetic_dataset, use_prompt=True, with_decode=False):

        self.synthetic_dataset = synthetic_dataset
        self.synthetic_labels = synthetic_dataset.get_labels()
        self.synthetic_data=self.process_extrat(self.synthetic_dataset,self.synthetic_labels,use_prompt, with_decode)
        random.seed(self.opt.seed)
        random.shuffle(self.synthetic_data)
        return self.process_synthe(self.synthetic_data, with_decode=with_decode)

    def process_synthetic_data_rp(self, synthetic_dataset, use_prompt=True, with_decode=False):

        self.synthetic_dataset = synthetic_dataset
        self.synthetic_labels = synthetic_dataset.get_labels()
        self.synthetic_data=self.process_extrat_rp(self.synthetic_dataset,self.synthetic_labels)
        random.seed(self.opt.seed)
        random.shuffle(self.synthetic_data)
        return self.process_synthe(self.synthetic_data)

    def process_synthe(self,data,method='No', sample_num=300):
        if method =='ST':
            temp_dataloader = DataLoader(data,
                                       batch_size=1,
                                       num_workers=1,
                                       drop_last=False,
                                       shuffle=False,
                                       collate_fn=collate_fn_extract)
            self.model.eval()
            bar = tqdm(temp_dataloader)
            data_perp = {}
            with torch.no_grad():
                for i, batch in enumerate(bar):
                    batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    outputs = self.step(batch)
                    data[i]['perp'] = torch.exp(outputs.loss).cpu().detach().tolist()

            if not Path(self.opt.output_path + 'process_synthetic.jsonl').exists():
                for sent in self.synthetic_dataset.sents:
                    for trip in sent.triplets:
                        trip.score=data[i]['perp']
                self.synthetic_dataset.save(self.opt.output_path+'process_synthetic.jsonl')

            self.synthetic_dataloader = DataLoader(data,
                                                   batch_size=self.opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=False,
                                                   shuffle=True,
                                                   collate_fn=collate_fn_extract)
        # elif method =='inter':
        #     pred_labels = data.get_labels()
        #     prompt = 'Candidate relation: ' + ', '.join(pred_labels) + '. '
        #     texts = [s.text for s in data.sents]
        #     gen = TextGenerator(
        #         model=self.model,
        #         tokenizer=T5Tokenizer.from_pretrained(model_path),
        #         max_length=128,
        #     )
        #     gen.model = gen.model.to(device)
        #     constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        #     sents = []
        #
        #     for i in tqdm(range(0, len(texts), self.opt.batch_size)):
        #         batch = texts[i: i + self.opt.batch_size]
        #         x = [prompt + self.encoder.encode_x(t) for t in batch]
        #
        #         outputs = gen.run(
        #             x, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
        #         )
        #         assert len(outputs) == len(x)
        #
        #         for i, raw in enumerate(outputs):
        #             triplet = self.encoder.safe_decode(x[i], y=raw)
        #             if use_label_constraint:
        #                 assert gen.scores is not None
        #                 triplet = constraint.run(triplet, gen.scores[i])
        #             sents.append(Sentence(triplets=[triplet]))
        #
        #     Dataset(sents=sents).save(path_out)
        #     temp_dataloader = DataLoader(data,
        #                                batch_size=1,
        #                                num_workers=1,
        #                                drop_last=False,
        #                                shuffle=False,
        #                                collate_fn=collate_fn_extract)
        #     self.model.eval()
        #     bar = tqdm(temp_dataloader)
        #     data_perp = {}
        #     with torch.no_grad():
        #         for i, batch in enumerate(bar):
        #             batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        #             outputs = self.step(batch, return_dict=True, output_hidden_states=True)
        #             data[i]['perp'] = torch.exp(outputs.loss).cpu().detach().tolist()
        #     data = sorted(data,key=lambda x:x['perp'])[:sample_num]
        #     self.synthetic_dataloader = DataLoader(data,
        #                                            batch_size=self.opt.batch_size,
        #                                            num_workers=4,
        #                                            drop_last=False,
        #                                            shuffle=True,
        #                                            collate_fn=collate_fn_extract)
        else:
            self.synthetic_dataloader = DataLoader(data,
                                                   batch_size=self.opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=False,
                                                   shuffle=True,
                                                   collate_fn=collate_fn_extract)
        return data
    def make_pipe(self, **kwargs) -> Pipeline:
        pipe = pipeline(
            self.pipe_name,
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            **kwargs,
        )
        return pipe

    def train_gen(self, model_path=None):
        if model_path is not None:
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            return None
        train_step, best_step, best_epoch, best_dev_per = 0, 0, 0, 1000
        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(self.train_dataloader)
            for i, batch in enumerate(bar):
                # wandb.log({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                train_step += 1
                batch = {k: v.to(self.opt.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.step(batch, return_dict=True, output_hidden_states=True, output_attentions=True)
                loss = outputs.loss
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            if self.opt.do_eval:
                perplexity, loss = self.evaluate_gen()
                print(f'Dev perplexity: {perplexity}, loss: {loss}')
                if perplexity<best_dev_per:
                    best_dev_per=perplexity
                    best_step = train_step
                    best_epoch = epoch
                    checkpoint_path = self.opt.output_path + 'generator/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{perplexity}/'
                    print(f"Save model {checkpoint_path}!")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                self.model.train()
            print(f'epoch:{epoch}.\n')
        perplexity, loss = self.evaluate_gen()
        print(f'Dev perplexity: {perplexity}, loss: {loss}')
        if perplexity < best_dev_per:
            best_dev_per = perplexity
            best_step = train_step
            best_epoch = epoch
            checkpoint_path = self.opt.output_path + 'generator/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{perplexity}/'
            print(f"Save model {checkpoint_path}!")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
        best_path = self.opt.output_path + 'generator/' +f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_per}/'
        if Path(self.opt.output_path + 'best_generator/').exists():
            shutil.rmtree(self.opt.output_path + 'best_generator/')
        shutil.move(best_path, self.opt.output_path + 'best_generator/')
        delete_checkpoints( self.opt.output_path + 'generator/')


    def evaluate_gen(self):
        with torch.no_grad():
            bar = tqdm(self.dev_dataloader)
            length = len(self.dev_dataloader)
            loss=0
            for i, batch in enumerate(bar):
                batch = {k: v.to(self.opt.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.step(batch)
                loss += outputs[0]
            perplexity = torch.exp(loss/length)
            return perplexity, loss

    def evaluate_extract(self):
        with torch.no_grad():
            self.model.eval()
            bar = tqdm(self.dev_dataloader)
            length = len(self.dev_dataloader)
            sentences, predicts, targets, score = [], [], [], 0
            for i, batch in enumerate(bar):
                if 'text_labels' in batch:
                    text_labels = batch['text_labels']
                else:
                    text_labels = None
                if 'context_pos' in batch:
                    batch_neg = {}
                    del batch['input_ids_con']
                    del batch['attention_mask_con']
                    del batch['input_ids_neg']
                    del batch['attention_mask_neg']

                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

                outputs = self.model.predict(**batch, return_dict=True, output_hidden_states=True)#
                score += outputs.loss
                score = score / self.opt.gradient_accumulation_steps
            score = score/length
        return score

    def evaluate_extract_weight(self):
        with torch.no_grad():
            self.model.eval()
            bar = tqdm(self.dev_dataloader)
            length = len(self.dev_dataloader)
            sentences, predicts, targets, score = [], [], [], 0
            for i, batch in enumerate(bar):
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

                outputs = self.step(batch)
                score += outputs.loss
                score = score / self.opt.gradient_accumulation_steps
            score = score/length
        return score

    def train_extract_weight(self, is_synthetic=False):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000
        if is_synthetic:
            data_loader = self.synthetic_dataloader
            self.opt.epochs = 10
        else:
            data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            for i, batch in enumerate(bar):
                # wandb.log({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                # weights_before = copy.deepcopy(self.model.state_dict())
                train_step += 1
                loss = 0
                #input_ids, attention_mask, decoder_attention_mask, support_idx, support_attn, support_decoder_attention_mask, support_labels
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                query_dict = {'support_ids':'input_ids', 'support_attn':'attention_mask', 'support_decoder_attention_mask':'decoder_attention_mask', 'support_labels':'labels'}
                # input_ds = support_ids
                # attention_mask = support_attn
                # labels = support_labels
                # decoder_attention_mask = support_decoder_attention_mask
                batch1 = {query_dict[k]: v for k, v in batch.items() if k in query_dict}
                batch2 = {k: v for k, v in batch.items() if k not in query_dict}
                outputs = self.step(batch2)
                loss += outputs.loss

                outputs = self.step(batch1)
                loss += outputs.loss
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()

                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(list(self.model.parameters()),
                                                   max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            if self.opt.do_eval: #  and (i+1) % len_data==0
                score = self.evaluate_extract_weight()
                checkpoint_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                print(f"Save model {checkpoint_path}!")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                if score < best_dev_score:
                    best_dev_score = score
                    best_epoch = epoch
                    best_step = train_step
        best_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if is_synthetic:
            if Path(self.opt.output_path + 'final_best_extractor/').exists():
                shutil.rmtree(self.opt.output_path + 'final_best_extractor/')
            shutil.move(best_path, self.opt.output_path + 'final_best_extractor/')
            delete_checkpoints(self.opt.output_path + 'extractor/')
            return self.opt.output_path + 'final_best_extractor/'
        else:
            if Path(self.opt.output_path + 'best_extractor/').exists():
                shutil.rmtree(self.opt.output_path + 'best_extractor/')
            shutil.move(best_path, self.opt.output_path + 'best_extractor/')
            delete_checkpoints(self.opt.output_path + 'extractor/')
            return self.opt.output_path + 'best_extractor/'

    def train_extract_rp(self, is_synthetic=False):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000
        if is_synthetic:
            self.opt.epochs = 3
            if not hasattr(self, 'optimizer'):
                self.process_extraction_data_rp()
            data_loader = self.synthetic_dataloader
        else:
            if not hasattr(self, 'optimizer'):
                self.process_extraction_data_rp()
            data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//3-1
            for i, batch in enumerate(bar):
                # wandb.log({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                # weights_before = copy.deepcopy(self.model.state_dict())
                train_step += 1
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # for j in range(self.opt.reptile_m):
                outputs = self.step(batch)
                loss = outputs.loss
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()

                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(list(self.model.parameters()),
                                                   max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                # weights_after = self.model.state_dict()

            if self.opt.do_eval: #  and (i+1) % len_data==0
                score = self.evaluate_extract()
                checkpoint_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                print(f"Save model {checkpoint_path}!")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                if score < best_dev_score:
                    best_dev_score = score
                    best_epoch = epoch
                    best_step = train_step

        best_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if is_synthetic:
            if Path(self.opt.output_path + 'final_best_extractor/').exists():
                shutil.rmtree(self.opt.output_path + 'final_best_extractor/')
            shutil.move(best_path, self.opt.output_path + 'final_best_extractor/')
            delete_checkpoints(self.opt.output_path + 'extractor/')
            return self.opt.output_path + 'final_best_extractor/'
        else:
            if Path(self.opt.output_path + 'best_extractor/').exists():
                shutil.rmtree(self.opt.output_path + 'best_extractor/')
            shutil.move(best_path, self.opt.output_path + 'best_extractor/')
            delete_checkpoints(self.opt.output_path + 'extractor/')
            return self.opt.output_path + 'best_extractor/'



    def train_extract_reptile(self, is_synthetic=False):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000
        if is_synthetic:
            data_loader = self.synthetic_dataloader
            self.opt.epochs = 10
        else:
            data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//3-1
            for i, batch in enumerate(bar):
                # wandb.log({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                # weights_before = copy.deepcopy(self.model.state_dict())
                train_step += 1
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # for j in range(self.opt.reptile_m):
                outputs = self.step(batch)
                loss = outputs.loss
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()

                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(list(self.model.parameters()),
                                                   max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                # weights_after = self.model.state_dict()

            if self.opt.do_eval: #  and (i+1) % len_data==0
                score = self.evaluate_extract()
                checkpoint_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                print(f"Save model {checkpoint_path}!")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                if score < best_dev_score:
                    best_dev_score = score
                    best_epoch = epoch
                    best_step = train_step

        best_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if is_synthetic:
            if Path(self.opt.output_path + 'final_best_extractor/').exists():
                shutil.rmtree(self.opt.output_path + 'final_best_extractor/')
            shutil.move(best_path, self.opt.output_path + 'final_best_extractor/')
            delete_checkpoints(self.opt.output_path + 'extractor/')
            return self.opt.output_path + 'final_best_extractor/'
        else:
            if Path(self.opt.output_path + 'best_extractor/').exists():
                shutil.rmtree(self.opt.output_path + 'best_extractor/')
            shutil.move(best_path, self.opt.output_path + 'best_extractor/')
            delete_checkpoints(self.opt.output_path + 'extractor/')
            return self.opt.output_path + 'best_extractor/'


    def train_extract_metric(self, is_synthetic=False, aux_loss_weight=0.):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000
        if is_synthetic:
            data_loader = self.synthetic_dataloader
            self.opt.epochs=10
        else:
            data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//4-1 #len(data_loader)=1

            # ipdb.set_trace()
            #在数据加载器上进行迭代，处理每个批次的数据:
            for i, batch in enumerate(bar):
                # wandb.log({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                loss = 0
                train_step += 1

                if 'text_labels' in batch:
                    text_labels = batch['text_labels']
                else:
                    text_labels = None
                if 'context_pos' in batch:
                    batch_con, batch_neg = {}, {}
                    context_pos = batch['context_pos']
                    batch_con['input_ids'] = batch['input_ids_con'].cuda()
                    batch_con['attention_mask'] = batch['attention_mask_con'].cuda()
                    batch_con['decoder_input_ids'] = None
                    batch_con['labels'] = batch['labels'].cuda()
                    context_pos_neg = batch['context_pos_neg']
                    batch_neg['input_ids'] = batch['input_ids_neg'].cuda()
                    batch_neg['attention_mask'] = batch['attention_mask_neg'].cuda()
                    batch_neg['decoder_input_ids'] = None
                    batch_neg['labels'] = batch['labels'].cuda()
                    del batch['input_ids_con']
                    del batch['attention_mask_con']
                    del batch['input_ids_neg']
                    del batch['attention_mask_neg']
                else:
                    batch_neg=None
                    context_pos, context_pos_neg = None, None

                if 'proto_pos' in batch:
                    proto_pos = torch.tensor(batch['proto_pos']).cuda()
                    proto_label = torch.tensor(batch['proto_label']).cuda()
                    # proto_pos = batch['proto_pos']
                else:
                    proto_pos, proto_label = None, None

                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                batch['aux_loss_weight']=aux_loss_weight
                outputs = self.step(batch)
                loss += outputs.loss

                # if proto_pos is not None:
                #     enc_hidden_states = outputs.encoder_hidden_states[-1]
                #     dec_hidden_states = outputs.decoder_hidden_states[-1]
                #     dec_attention_mask = batch['decoder_attention_mask']
                #     B = proto_pos.shape[0]
                #     hidden_states = get_proto_representation(proto_pos, enc_hidden_states, length=B)
                #     proto_emb = self.proto_fc(hidden_states)
                #     rel_marker = get_marker_state(batch['labels'], dec_hidden_states)
                #     logits = (proto_emb*rel_marker).sum(-1)
                #     loss+=F.cross_entropy(logits,proto_label)

                if 'generate_input_ids' in batch:
                    generate_batch, temp_batch={},{}
                    for gen_na in batch.keys():
                        if 'generate_' in gen_na:
                            generate_batch[gen_na.replace('generate_','')] = batch[gen_na]
                        else:
                            temp_batch[gen_na] = batch[gen_na]
                    generate_outputs = self.step(generate_batch, return_dict=True, output_hidden_states=True)
                    loss += 0.01*generate_outputs.loss
                    batch = temp_batch

                if context_pos is not None:
                    B = len(context_pos)
                    outputs = self.step(batch_con, return_dict=True, output_hidden_states=True)
                    decoder_context_hid = torch.stack([outputs.encoder_hidden_states[-1][i,pos,:] for i,pos in enumerate(context_pos)],dim=0).mean(1)#  B,D
                    outputs_neg = self.step(batch_neg, return_dict=True, output_hidden_states=True)
                    decoder_context_hid_neg = torch.stack(
                        [outputs_neg.encoder_hidden_states[-1][i, pos, :] for i, pos in enumerate(context_pos_neg)],
                        dim=0).mean(1)  # B,D

                    # temp_labels = torch.LongTensor(
                    #     [l for l in range(2) for b in range(B)]).cuda()
                    loss2 = nn.MSELoss()
                    distance_loss = -loss2(decoder_context_hid,decoder_context_hid_neg)
                    loss += 0.1*distance_loss
                    # inp = torch.cat([decoder_context_hid,decoder_context_hid_neg])
                    # loss += F.nll_loss(logits, temp_labels)

                # if text_labels is not None:
                #     neg_labels_num = 2
                #     labels_len = len(text_labels)
                #     target = torch.tensor([0]*labels_len).to(batch['input_ids'])
                #     self.relation_linear.to(loss)
                #     inp_ids_list = batch['input_ids'].tolist()
                #     output_ids_list = batch['labels'].tolist()
                #
                #     labels_idx = [self.tokenizer(text, add_special_tokens=False) for text in text_labels]
                #     inp_idx = [find_sublist_index(inp_ids,label) for inp_ids,label in zip(inp_ids_list,labels_idx)]
                #
                #     inp_neg_idx = [random.sample((torch.nonzero(batch['input_ids'][i]==6)[:self.opt.n_unseen].squeeze()+1).tolist(),neg_labels_num) for i in range(labels_len)]
                #     oup_idx = [find_sublist_index(output_ids,self.rel_prefix) for output_ids in output_ids_list]
                #
                #     relation_rep = self.relation_linear(outputs.encoder_hidden_states[-1])
                #     temp_rel = torch.stack([relation_rep[i,inp] for i,inp in enumerate(inp_idx)])
                #     temp_rel_neg = torch.stack([outputs.encoder_hidden_states[-1][i,neg] for i,inp_neg in enumerate(inp_neg_idx) for neg in inp_neg])
                #     temp_rel_neg = temp_rel_neg.reshape(labels_len,neg_labels_num,-1)
                #     temp_rel = torch.cat([temp_rel.unsqueeze(1),temp_rel_neg],1)
                #     temp_pre = torch.stack([outputs.decoder_hidden_states[-1][i,oup] for i,oup in enumerate(oup_idx)])
                #     pred = torch.einsum('bik,bk->bi', temp_rel, temp_pre)
                #     loss += rel_loss_weight * self.relation_loss(pred,target)

                # outputs.decoder_hidden_states[-1]
                # bar.set_description(f"EPOCH {str(epoch).zfill(2)} --> loss: {format(loss.item(), '.4f')}")
                # wandb.log({'loss': loss.item()})
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                if train_step % self.opt.gradient_accumulation_steps == 0:
                    # torch.nn.utils.clip_grad_norm_(list(self.model.parameters())+list(self.neg_cls.parameters()), max_norm=self.opt.gradient_clip_val)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
                    # torch.nn.utils.clip_grad_norm_(self.neg_cls.parameters(), max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

                if self.opt.do_eval and (i+1) % len_data==0: #  and (i+1) % len_data==0
                    score = self.evaluate_extract()
                    checkpoint_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                    print(f"Save model {checkpoint_path}!")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    if score < best_dev_score:
                        best_dev_score = score
                        best_epoch = epoch
                        best_step = train_step

        best_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if is_synthetic:
            if Path(self.opt.output_path + 'final_best_extractor/').exists():
                shutil.rmtree(self.opt.output_path + 'final_best_extractor/')
            shutil.move(best_path, self.opt.output_path + 'final_best_extractor/')
            delete_checkpoints(self.opt.output_path + 'extractor/')
            return self.opt.output_path + 'final_best_extractor/'
        else:
            if Path(self.opt.output_path + 'best_extractor/').exists():
                shutil.rmtree(self.opt.output_path + 'best_extractor/')
            shutil.move(best_path, self.opt.output_path + 'best_extractor/')
            delete_checkpoints(self.opt.output_path + 'extractor/')
            return self.opt.output_path + 'best_extractor/'

    def train_extract_with_synthetic(self, is_final=True):
        self.opt.epochs=2
        if not hasattr(self, 'optimizer'):
            self.process_extraction_data_rp()

        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000
        data_loader = self.synthetic_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//3-1
            for i, batch in enumerate(bar):
                # wandb.log({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                train_step += 1

                if 'text_labels' in batch:
                    text_labels = batch['text_labels']
                else:
                    text_labels = None

                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.step(batch)
                loss = outputs.loss
                # if text_labels is not None:
                #     neg_labels_num = 2
                #     labels_len = len(text_labels)
                #     target = torch.tensor([0]*labels_len).to(batch['input_ids'])
                #     self.relation_linear.to(loss)
                #     inp_ids_list = batch['input_ids'].tolist()
                #     output_ids_list = batch['labels'].tolist()
                #
                #     labels_idx = [self.tokenizer(text, add_special_tokens=False) for text in text_labels]
                #     inp_idx = [find_sublist_index(inp_ids,label) for inp_ids,label in zip(inp_ids_list,labels_idx)]
                #
                #     inp_neg_idx = [random.sample((torch.nonzero(batch['input_ids'][i]==6)[:self.opt.n_unseen].squeeze()+1).tolist(),neg_labels_num) for i in range(labels_len)]
                #     oup_idx = [find_sublist_index(output_ids,self.rel_prefix) for output_ids in output_ids_list]
                #
                #     relation_rep = self.relation_linear(outputs.encoder_hidden_states[-1])
                #     temp_rel = torch.stack([relation_rep[i,inp] for i,inp in enumerate(inp_idx)])
                #     temp_rel_neg = torch.stack([outputs.encoder_hidden_states[-1][i,neg] for i,inp_neg in enumerate(inp_neg_idx) for neg in inp_neg])
                #     temp_rel_neg = temp_rel_neg.reshape(labels_len,neg_labels_num,-1)
                #     temp_rel = torch.cat([temp_rel.unsqueeze(1),temp_rel_neg],1)
                #     temp_pre = torch.stack([outputs.decoder_hidden_states[-1][i,oup] for i,oup in enumerate(oup_idx)])
                #     pred = torch.einsum('bik,bk->bi', temp_rel, temp_pre)
                #     loss += rel_loss_weight * self.relation_loss(pred,target)

                # outputs.decoder_hidden_states[-1]
                # bar.set_description(f"EPOCH {str(epoch).zfill(2)} --> loss: {format(loss.item(), '.4f')}")
                # wandb.log({'loss': loss.item()})
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

                if self.opt.do_eval and (i+1) % len_data==0: #
                    score = self.evaluate_extract()
                    checkpoint_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                    print(f"Save model {checkpoint_path}!")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    if score < best_dev_score:
                        best_dev_score = score
                        best_epoch = epoch
                        best_step = train_step

        best_path = self.opt.output_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'

        if Path(self.opt.output_path + 'final_best_extractor/').exists():
            shutil.rmtree(self.opt.output_path + 'final_best_extractor/')
        if is_final:
            model_path = self.opt.output_path + 'final_best_extractor/'
        else:
            model_path = self.opt.output_path + 'best_extractor/'

        shutil.move(best_path, model_path)
        delete_checkpoints(self.opt.output_path + 'extractor/')
        return model_path


    def predict(self, path_in, path_out, model_path, use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        prompt = '[PROTO] '+', '.join(pred_labels)+'. '
        texts = [s.text for s in data.sents]
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        # constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        constraint = LabelConstraint_rp(labels=pred_labels, tokenizer=self.tokenizer)
        sents = []

        for i in tqdm(range(0, len(texts), self.opt.batch_size)):
            batch = texts[i: i + self.opt.batch_size]
            x = [prompt + self.encoder.encode_x(t) for t in batch]

            outputs = gen.run(
                x, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )
            assert len(outputs) == len(x)

            for i, raw in enumerate(outputs):
                triplet = self.encoder.safe_decode(x[i], y=raw)
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[i])
                sents.append(Sentence(triplets=[triplet]))

        Dataset(sents=sents).save(path_out)

    def predict_rc(self, path_in, path_out, model_path, use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        prompt = '[PROTO] '+', '.join(pred_labels)+'. '
        texts, ht_en_texts=[],[]
        for s in data.sents:
            for t in s.triplets:
                tokens = t.tokens
                head_start = t.head[0]
                head_end = t.head[-1]+1
                tail_start = t.head[0]
                tail_end = t.head[-1]+1
                texts.append(s.text)
                ht_en_texts.append('[HEAD] '+ ' '.join(tokens[head_start:head_end]) +', [TAIL] '+ ' '.join(tokens[tail_start:tail_end]) + ' .')
        # texts = [s.text for s in data.sents for t in s.triplets]
        ori_triplets = [t for s in data.sents for t in s.triplets]
        assert len(ori_triplets)==len(texts)
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents = []
        preds, golds = [], []
        for i in tqdm(range(0, len(texts), self.opt.predict_batch_size)):
            batch = texts[i: i + self.opt.predict_batch_size]
            batch_ht = ht_en_texts[i: i + self.opt.predict_batch_size]
            ori_triplet = ori_triplets[i: i + self.opt.predict_batch_size]
            x = [prompt + ht + self.encoder.encode_x(t) for t,ht in zip(batch,batch_ht)]
            # decoder_input_ids = pad_sequence([self.tokenizer(ht_en, return_tensors='pt', add_special_tokens=False).input_ids[0] for ht_en in batch_ht],batch_first=True)
            # decoder_attention_mask = pad_sequence([self.tokenizer(ht_en, return_tensors='pt', add_special_tokens=False).attention_mask[0] for ht_en in batch_ht],batch_first=True)
            outputs = gen.run(
                x, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )

            assert len(outputs) == len(x)
            for i, raw in enumerate(outputs):
                triplet = self.encoder.safe_decode(x[i], y=raw)
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[i])
                # sents.append(Sentence(triplets=[triplet]))
                preds.append(triplet.label)
            golds.extend([trip.label for trip in ori_triplet])

        # Dataset(sents=sents).save(path_out)
        score_rc(preds,golds)


    def predict_rp(self, path_in, path_out, model_path, use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        prompt = '[PROTO] '+', '.join(pred_labels)+'. '
        texts = [s.text for s in data.sents]
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint_rp(labels=pred_labels, tokenizer=self.tokenizer)
        sents = []

        for i in tqdm(range(0, len(texts), self.opt.batch_size)):
            batch = texts[i: i + self.opt.batch_size]
            x = ['' + self.encoder.encode_x(t) for t in batch] # prompt

            outputs = gen.run(
                x, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )
            assert len(outputs) == len(x)

            for i, raw in enumerate(outputs):
                triplet = self.encoder.safe_decode(x[i], y=raw)
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[i])
                sents.append(Sentence(triplets=[triplet]))

        Dataset(sents=sents).save(path_out)

    def predict_weight(self, path_in, path_out, model_path, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        prompt = '[PROTO1] [PROTO2] [PROTO3]' + ', '.join(pred_labels)+ '. '
        texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_extrat_weight_predict(data,pred_labels, prompt=prompt)
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=self.opt.batch_size,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_extract_weight_predict)
        # prompt = '[REL] '+', '.join(pred_labels)+'. '
        #
        # temp_prompt = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
        # prompt_len = temp_prompt['input_ids'].shape[1] - 1
        # texts = [s.text for s in data.sents]
        gen = TextGenerator_weight(
            model=T5ForConditionalGenerationWithAdapter.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents, sents_no = [], 0
        for batch in tqdm(self.test_dataloader):
            del batch['labels']
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen.run(
                batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )

            for j, raw in enumerate(outputs):
                triplet = self.encoder.safe_decode(texts[sents_no], y=raw)
                sents_no+=1
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[j])
                sents.append(Sentence(triplets=[triplet]))

        Dataset(sents=sents).save(path_out)

    def predict_multi(self, path_in: str, path_out: str, model_path=None, use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        prompt = '[PROTO] ' + ', '.join(pred_labels) + '. '
        texts = [s.text for s in data.sents]
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents = []

        for i in tqdm(range(0, len(texts), self.opt.batch_size)):
            batch = texts[i: i + self.opt.batch_size]
            x = [prompt + self.encoder.encode_x(t) for t in batch]

            outputs = gen.run(
                x, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )
            assert len(outputs) == len(x)

            for j, raw in enumerate(outputs):
                triplets=[]
                if len(raw.split('. [HEAD]'))>2:
                    raw_list = raw.split('. [HEAD]')
                    for raw in raw_list:
                        if '[HEAD]' not in raw:
                            triplet = self.encoder.safe_decode(x[j], y='[HEAD]' + raw + ' .')
                        else:
                            triplet = self.encoder.safe_decode(x[j], y=raw + ' .')
                        if use_label_constraint:
                            assert gen.scores is not None
                            triplet = constraint.run(triplet, gen.scores[j])
                        triplets.append(triplet)
                else:
                    triplet = self.encoder.safe_decode(x[j], y=raw)
                    if use_label_constraint:
                        assert gen.scores is not None
                        triplet = constraint.run(triplet, gen.scores[j])
                    triplets.append(triplet)
                sents.append(Sentence(triplets=triplets))

        Dataset(sents=sents).save(path_out)

    def predict_multi_ori(self, path_in: str, path_out: str, model_path=None, max_target_length=128, search_threshold=-0.9906):
        stem = Path(path_out).stem
        path_raw = path_out.replace(stem, f"{stem}_raw")
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        prompt = '[PROTO] ' + ', '.join(pred_labels) + '. '
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=gen.tokenizer)
        searcher = TripletSearchDecoder(
            gen=gen, encoder=self.encoder, constraint=constraint, prompt=prompt
        )

        sents = [
            Sentence(tokens=s.tokens, triplets=searcher.run(s.text))
            for s in tqdm(data.sents)
        ]

        Dataset(sents=sents).save(path_raw)
        for s in sents:
            s.triplets = [t for t in s.triplets if t.score > search_threshold]
        Dataset(sents=sents).save(path_out)

    def load_extract_model(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        if torch.cuda.is_available():
            self.model.cuda()


    def extract(self, input_ids, attention_mask, num_beams=1, max_len=256):
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            num_beams=num_beams,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        # decoder_hidden_state = torch.cat([x[-1] for x in out['decoder_hidden_states']], dim=1)
        # pred_seq = out.sequences[:, 1:]
        return out.sequences

    def generate(self, relation, encoder, pipe, num=250, max_len=128, batch_size=64):
        prompt = encoder.encode_x(relation)
        sents, raw = [], []
        errors = set()

        while len(sents) < num:
            outputs = pipe(
                [prompt],
                num_return_sequences=batch_size,
                max_length=max_len,
            )
            for o in outputs:
                raw.append(o["generated_text"] + "\n")
                x, y = encoder.parse_line(raw[-1])  # 生成格式符合标准。
                try:
                    s = encoder.decode(x=prompt, y=y)
                    if s.is_valid():
                        sents.append(s)
                except Exception as e:
                    errors.add(str(e))

            print(dict(target=num, success=len(sents), raw=len(raw)))

        assert len(sents) >= num
        print(dict(prompt=prompt, success_rate=len(sents) / len(raw), errors=errors))
        return sents[:num], raw

    def step(self, batch):
        outputs = self.model(**batch)
        return outputs

    def get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        other_lr_name = ['adapter']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in other_lr_name)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and not any(nd in n for nd in other_lr_name)],
                "weight_decay": 0.0,
                "lr": self.opt.lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in other_lr_name)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.aux_lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and any(nd in n for nd in other_lr_name)],
                "weight_decay": 0.0,
                "lr": self.opt.aux_lr
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)
        t_total = (len(self.train_dataset.sents) // self.opt.batch_size // self.opt.gradient_accumulation_steps * float(self.opt.epochs))
        warmup_steps = int(self.opt.warmup_ratio * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total) if self.opt.use_scheduler else None
        return optimizer, scheduler