import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config
from torch.nn.utils.rnn import pad_sequence
from transformers.models import gpt2
import torch
from pydantic import BaseModel
import torch.nn.functional as F
import ipdb
from typing import Dict, List, Optional, Tuple

def find_sublist_index(items: list, query: list):
    length = len(query)
    for i in range(len(items) - length + 1):
        if items[i : i + length] == query:
            return i
    return -1

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

class DynamicModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

def get_word_representation(matrix, states):  # word_index, hidden_states
    length = matrix.size(1)
    min_value = torch.min(states).item()
    states = states.unsqueeze(1).expand(-1, length, -1, -1)
    states = torch.masked_fill(states, matrix.eq(0).unsqueeze(-1), min_value)
    word_reps, _ = torch.max(states, dim=2)
    # word_reps = torch.relu(F.dropout(word_reps, p=0.1, training=self.training))
    return word_reps


def get_marker_state(index, marker_position, decoder_hidden_state):
    assert index in [1, 2, 3]  # 1-aspect, 2-opinion, 3-sentiment
    aos_marker_position = marker_position.eq(index)
    aos_marker_len = torch.sum(aos_marker_position, dim=-1).unsqueeze(-1)
    aos_marker = decoder_hidden_state[aos_marker_position.bool()]
    return aos_marker, aos_marker_len

class BaseT5(nn.Module):
    def __init__(self, args):
        super(BaseT5, self).__init__()
        self.args = args
        self.config = T5Config.from_pretrained(self.args.t5_pretrain_model_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(self.args.t5_pretrain_model_path, config=self.config)

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask):
        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.loss, None

    def predict(self, input_ids, attention_mask):
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.max_len,
            num_beams=self.args.num_beams,
        )

class BaseGPT(nn.Module):
    def __init__(self,args):
        super(BaseGPT, self).__init__()
        self.args = args
        self.config = GPT2Config.from_pretrained(self.args.gpt2_pretrain_model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", config=self.config)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=self.config)

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.loss, None

    def predict(self, input_ids, attention_mask):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.max_len,
            num_beams=self.args.num_beams,
        )

class ProtoT5(nn.Module):
    def __init__(self, args, tokenizer, t5_model, encoder, use_proto=True):
        super(ProtoT5, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.config = T5Config.from_pretrained(self.args.t5_pretrain_model_path)
        self.t5 = t5_model
        self.encoder = encoder
        self.leak_relu = torch.nn.LeakyReLU(0.1)
        self.ignore_index = -100
        if use_proto:
            self.proj_head = nn.Linear(2 * self.config.d_model, self.config.d_model, bias=True)
            self.classifier_head = nn.Linear(self.config.d_model, 2, bias=True)
            self.proj_tail = nn.Linear(2 * self.config.d_model, self.config.d_model, bias=True)
            self.classifier_tail = nn.Linear(self.config.d_model, 2, bias=True)
            self.proj_rel = nn.Linear(2 * self.config.d_model, self.config.d_model, bias=True)
            self.classifier_rel = nn.Linear(self.config.d_model, 2, bias=True)

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask, decoder_input_ids=None, rel_pos=None, rel_label_int=None, rel_dec_proto=None,
                head_pos=None, head_dec_proto=None, tail_pos=None, tail_dec_proto=None, aux_loss_weight=0.1, **kwargs):
        # ipdb.set_trace()
        outputs = self.t5(
            input_ids, #'[PROTO] religion, religion. Context : In 1689, Konstanty was one of the judges who sentenced Kazimierz <unk> yszczy<unk> ski to death for atheism.</s>'
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,#空
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,#'Head Entity : Kazimierz <unk> yszczy<unk> ski, Tail Entity : atheism, Relation : religion.</s>'
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            **kwargs
        )
        #outputs.keys():odict_keys(['loss', 'logits', 'past_key_values', 'decoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions']
        enc_hidden_states = outputs.encoder_hidden_states[-1]
        dec_hidden_states = outputs.decoder_hidden_states[-1]
        if rel_pos is not None and aux_loss_weight>0.:
            B,rel_num,_ = rel_pos.shape
            rel_pos = rel_pos.reshape(-1,2)
            temp_label = torch.tensor([1]*rel_pos.shape[0]).cuda()
            # rel_proto = torch.stack([torch.stack([torch.mean(enc_hidden_states[i,rel_pos[i,j,0]:rel_pos[i,j,1]],dim=0) for j in range(rel_num)]) for i in range(B)])
            rel_proto = torch.stack([torch.max(enc_hidden_states[i,rel_pos[i,0]:rel_pos[i,1]],dim=0)[0] for i in range(B*rel_num)])
            dec_proto = torch.stack([dec_hidden_states[i,rel_dec_proto[i,0]] for i in range(B) for j in range(rel_num)])

            hidden_states = torch.cat([rel_proto,dec_proto],dim=-1)
            # states = self.leak_relu(self.proj_fc(hidden_states))
            states = torch.relu(self.proj_rel(hidden_states))
            logits = self.classifier_rel(states)

            aux_proto_loss = torch.sum(F.cross_entropy(logits.view(-1, 2), temp_label, reduction='none'))
            outputs.loss += aux_loss_weight * aux_proto_loss

        if head_pos is not None and aux_loss_weight>0.:
            B,head_num,_ = head_pos.shape
            temp_label = torch.tensor([1]*B*head_num*2).cuda()

            head_proto = torch.stack([torch.max(enc_hidden_states[i*head_num+j,head_pos[i,j,0]:head_pos[i,j,1]],dim=0)[0] for i in range(B) for j in range(head_num)])
            # head_proto = torch.stack([torch.max(enc_hidden_states[i+j,head_pos[i,j,0]:head_pos[i,j,1]],dim=0) for i in range(B) for j in range(head_num)])
            tail_proto = torch.stack([torch.max(enc_hidden_states[i*head_num+j,tail_pos[i,j,0]:tail_pos[i,j,1]],dim=0)[0] for i in range(B) for j in range(head_num)])
            # tail_proto = torch.stack([torch.max(enc_hidden_states[i,tail_pos[i,0]:tail_pos[i,1]],dim=0)[0] for i in range(B)])

            head_dec_proto = torch.stack([dec_hidden_states[i*head_num+j,head_dec_proto[i,0]] for i in range(B) for j in range(head_num)])
            tail_dec_proto = torch.stack([dec_hidden_states[i*head_num+j,tail_dec_proto[i,0]] for i in range(B) for j in range(head_num)])

            head_hidden_states = torch.cat([head_proto,head_dec_proto],dim=-1)
            tail_hidden_states = torch.cat([tail_proto,tail_dec_proto],dim=-1)
            hidden_states = torch.stack([head_hidden_states,tail_hidden_states],dim=0)
            states = torch.relu(self.proj_head(hidden_states))
            logits = self.classifier_head(states)

            # states_head = torch.relu(self.proj_head(head_hidden_states))
            # logits_head = self.classifier_head(states_head)
            # states_tail = torch.relu(self.proj_tail(tail_hidden_states))
            # logits_tail = self.classifier_tail(states_tail)
            # logits = torch.stack([logits_head,logits_tail],dim=-1)

            aux_proto_loss = torch.sum(F.cross_entropy(logits.view(-1,2), temp_label, reduction='none'))

            outputs.loss += aux_loss_weight*aux_proto_loss

        # B = rel_pos.shape[0]

        # hidden_states = get_proto_representation(proto_pos, enc_hidden_states, length=B)
        # proto_emb = self.proto_fc(hidden_states)
        # rel_marker = get_marker_state(batch['labels'], dec_hidden_states)
        # logits = (proto_emb*rel_marker).sum(-1)
        # loss+=F.cross_entropy(logits,proto_label)
        return outputs

    def predict(self, input_ids, attention_mask, labels, decoder_attention_mask, decoder_input_ids=None, rel_pos=None, rel_label_int=None, rel_dec_proto=None,
                **kwargs):
        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        # if rel_pos is not None:
        #     B, rel_num = rel_pos.shape[0], rel_pos.shape[1]
        #     enc_hidden_states = outputs.encoder_hidden_states[-1]
        #     dec_hidden_states = outputs.decoder_hidden_states[-1]
        #     rel_proto = torch.stack([torch.stack(
        #         [torch.mean(enc_hidden_states[i, rel_pos[i, j, 0]:rel_pos[i, j, 1]], dim=0) for j in range(rel_num)])
        #                              for i in range(B)])
        return outputs

    def generate(self, input_ids, attention_mask, labels, decoder_attention_mask, decoder_input_ids=None, rel_pos=None, rel_label_int=None, rel_dec_proto=None,
                 **kwargs):
        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
        return outputs

    def save_pretrained(self,checkpoint_path):
        self.t5.save_pretrained(checkpoint_path)