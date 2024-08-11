import json
from pathlib import Path
from typing import Dict, List, Tuple

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer
from zsre_dataset import RelationData, RelationSentence

def encode_to_line(x: str, y: str) -> str:
    # Refer to original transformers readme
    text = json.dumps(dict(text=x, summary=y)) + "\n"
    assert decode_from_line(text) == (x, y)
    return text


def decode_from_line(text: str) -> Tuple[str, str]:
    d = json.loads(text)
    return d["text"], d["summary"]

class Encoder(BaseModel):
    def encode_x(self, x: str) -> str:
        raise NotImplementedError

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        raise NotImplementedError


    def decode(self, x: str, y: str) -> RelationSentence:
        raise NotImplementedError

    def decode_x(self, x: str) -> str:
        raise NotImplementedError

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def encode_to_line(self, sent: RelationSentence) -> str:
        raise NotImplementedError

    def encode_to_line_with_keyword(self, sent: RelationSentence) -> str:
        raise NotImplementedError

    def decode_from_line(self, line: str) -> RelationSentence:
        raise NotImplementedError

    def parse_line(self, line: str) -> Tuple[str, str]:
        raise NotImplementedError


class GenerateEncoder(Encoder):
    def encode_x(self, r: str) -> str:
        return f"Relation : {r} ."

    def decode_x(self, text: str) -> str:
        return text.split("Relation : ")[-1][:-2]

    def encode_triplet(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Context : {sent.text} Head Entity : {s} , Tail Entity : {o} ."

    def decode_triplet(self, text: str, label: str) -> RelationSentence:
        key_word=None
        front, back = text.split(" Head Entity : ")
        _, context = front.split("Context : ")
        head, back = back.split(" , Tail Entity : ")
        if 'Keyword :' in back:
            back, key_word = back.split(' Keyword :')
            context = context+ ' Keyword :'+ key_word  #Test
            key_word=key_word[:-2].strip().split(' ')
        else:
            key_word = ['']
        tail = back[:-2]
        return RelationSentence.from_spans(context, head, tail, label, keyword=key_word)

    def encode_y(self, sent: RelationSentence) -> str:
        return self.encode_x(sent.label) + " " + self.encode_triplet(sent)

    def decode_y(self, text: str, label: str) -> RelationSentence:
        del label
        front, back = text.split(". Context : ")
        label = self.decode_x(front + " .")
        return self.decode_triplet("Context : " + back, label)

    def encode_keyword(self, r_list: List[str]) -> str:
        r= ' '.join(r_list)
        return f" Keyword : {r} ."

    def decode(self, x: str, y: str) -> RelationSentence:
        r = self.decode_x(x)
        sent = self.decode_y(y, r)
        return sent

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.label)
        y = self.encode_y(sent)
        return x, y

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return y + "\n"

    def encode_to_line_with_keyword(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        keyword = self.encode_keyword(sent.keyword)
        return y + keyword + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        return "", line.strip()

class ExtractEncoder(Encoder):
    def encode_x(self, text: str) -> str:
        return f"[SENT] : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("[SENT] : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"[HEAD] {s} , [TAIL] {o} , [REL] {r} ."

    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(", [REL] ")
        label = label.strip()[:-2]
        front, tail = front.split(", [TAIL] ")
        _, head = front.split("[HEAD] ")
        return RelationSentence.from_spans(context, head.strip(), tail.strip(), label.strip())

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"[HEAD] {head} , [TAIL] {tail} , [REL] "

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return decode_from_line(line)

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

class ExtractEncoder_rp(Encoder):
    def encode_x(self, text: str) -> str:
        return f"Context : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("Context : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Head Entity : {s} , Tail Entity : {o} , Relation : {r} ."


    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(", Relation : ")
        label = label[:-2]
        front, tail = front.split(", Tail Entity : ")
        _, head = front.split("Head Entity : ")
        return RelationSentence.from_spans(context, head.strip(), tail.strip(), label.strip())

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"Head Entity : {head} , Tail Entity : {tail} , Relation :"

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return decode_from_line(line)

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

class ExtractEncoder_R1st(Encoder):
    def encode_x(self, text: str) -> str:
        return f"Context : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("Context : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Relation : {r} , Head Entity : {s} , Tail Entity : {o} ."

    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, tail = y.split(" , Tail Entity : ")
        tail=tail.strip()[:-2]
        front, head = front.split(", Head Entity : ")
        _, label = front.split("Relation : ")
        return RelationSentence.from_spans(context, head.strip(), tail.strip(), label.strip())

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"Head Entity : {head} , Tail Entity : {tail} , Relation :"

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return decode_from_line(line)

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

class ExtractEncoder_T1st(Encoder):
    def encode_x(self, text: str) -> str:
        return f"Context : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("Context : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Tail Entity : {o} , Head Entity : {s} , Relation : {r} ."

    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(" , Relation : ")
        tail=label.strip()[:-2]
        front, head = front.split(", Head Entity : ")
        _, tail = front.split("Tail Entity : ")
        return RelationSentence.from_spans(context, head.strip(), tail.strip(), label.strip())

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"Tail Entity : {tail} , Head Entity : {head} , Relation :"

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return decode_from_line(line)

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

def select_encoder(name: str) -> Encoder:
    mapping: Dict[str, Encoder] = dict(
        extract=ExtractEncoder(),
        extract_t1st=ExtractEncoder_T1st(),
        extract_rp=ExtractEncoder_rp(),
        generate=GenerateEncoder(),
    )
    encoder = mapping[name]
    return encoder


def test_entity_prompts(
    path: str = "outputs/data/zsl/wiki/unseen_10_seed_0/test.jsonl", limit: int = 100
):
    def tokenize(text: str, tok) -> List[str]:
        return tok.convert_ids_to_tokens(tok(text, add_special_tokens=False).input_ids)

    data = RelationData.load(Path(path))
    e = ExtractEncoder()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    print(tokenizer)
    for i, s in enumerate(tqdm(data.sents[:limit])):
        head, label, tail = s.as_tuple()
        x, y = e.encode(s)
        prompt = e.encode_entity_prompt(head, tail)
        tokens_y = tokenize(y, tokenizer)
        tokens_prompt = tokenize(prompt, tokenizer)
        assert tokens_y[: len(tokens_prompt)] == tokens_prompt
        if i < 3:
            print(tokens_y)


if __name__ == "__main__":
    Fire()
