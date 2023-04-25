import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, GPT2Tokenizer
from utils import pad_collate

# Tokenizer Arguments : https://stackoverflow.com/questions/65246703/how-does-max-length-padding-and-truncation-arguments-work-in-huggingface-bertt

class YelpDataset(Dataset):
    def __init__(self, datadir, label1_file, label0_file, enc_tokenizer, dec_tokenizer):       
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.dec_bos_token = '<|endoftext|> '
        self.dec_eos_token = '<|endoftext|>'
        self.dec_sep_token = ' ## '
        self.labels = []
        self.sentences = []
        self.num_classes = 2
        self.label_prompt = {0: 'negative', 1: 'positive'}

        positive_path = os.path.join(datadir, label1_file)
        negative_path = os.path.join(datadir, label0_file)
        with open(positive_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip().split('\n')
                self.labels.append(int(1))
                self.sentences.append(sentence)
        with open(negative_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip().split('\n')
                self.labels.append(int(0))
                self.sentences.append(sentence)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence, label = self.sentences[idx], self.labels[idx]
        enc_tokenized = self.enc_tokenizer(sentence, add_special_tokens=True, return_tensors='pt')
        if self.dec_tokenizer:
            dec_sentence = [self.dec_bos_token + s + self.dec_eos_token for s in sentence]
            dec_sentence_wprompt = [self.dec_bos_token + self.label_prompt[label] + self.dec_sep_token + s + self.dec_eos_token for s in sentence]
            
            dec_tokenized = self.dec_tokenizer(dec_sentence, add_special_tokens=True, return_tensors="pt")
            dec_tokenized_wprompt = self.dec_tokenizer(dec_sentence_wprompt, add_special_tokens=True, return_tensors="pt")

        else:
            dec_tokenized = {'input_ids': torch.tensor([[]]), 'attention_mask': torch.tensor([[]])}
            dec_tokenized_wprompt = {'input_ids': torch.tensor([[]]), 'attention_mask': torch.tensor([[]])}

        label = torch.tensor(label)
        label = torch.nn.functional.one_hot(label.to(torch.int64), self.num_classes)

        return {'enc_input_ids': enc_tokenized['input_ids'].squeeze(), 'enc_attention_mask': enc_tokenized['attention_mask'].squeeze(), 
                'dec_input_ids': dec_tokenized['input_ids'].squeeze(), 'dec_attention_mask': dec_tokenized['attention_mask'].squeeze(), 
                'dec_input_ids_wprompt': dec_tokenized_wprompt['input_ids'].squeeze(), 'dec_attention_mask_wprompt': dec_tokenized_wprompt['attention_mask'].squeeze(), 
                'labels': label}

def testing_yelploader():
    encoder_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", eos_token='<|endoftext|>', pad_token='<|endoftext|>')

    yelp_path = "/local1/pshah7/insnet/data/yelp/raw"

    train_dataset = YelpDataset(yelp_path, 'sentiment.train.1', 'sentiment.train.0', encoder_tokenizer, decoder_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)

    return next(iter(train_dataloader))

if __name__ == "__main__":
    
    yelp_batch = testing_yelploader()
    
    from IPython import embed; embed(); exit()