import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, GPT2Tokenizer

# Tokenizer Arguments : https://stackoverflow.com/questions/65246703/how-does-max-length-padding-and-truncation-arguments-work-in-huggingface-bertt

class YelpDataset(Dataset):
    def __init__(self, datadir, label1_file, label0_file, enc_tokenizer, dec_tokenizer):       
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.dec_bos_token = '<|endoftext|> '
        self.dec_eos_token = ' ## '
        # self.dec_eos_token = ' <|endoftext|>'
        # self.dec_sep_token = ' <|sep_token|> '
        self.dec_sep_token = ' <|endoftext|> '
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

        self.max_enc_tokenseq_length = max([len(self.enc_tokenizer(text)['input_ids'][0]) for text in self.sentences])
        self.max_dec_tokenseq_length = max([len(self.dec_tokenizer(text)['input_ids'][0]) for text in self.sentences])

        # print('Longest tokenized sequence for the encoder tokenizer: ' + str(self.max_enc_tokenseq_length))
        # print('Longest tokenized sequence for the decoder tokenizer: ' + str(self.max_dec_tokenseq_length))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        
        sentence, label = self.sentences[idx], self.labels[idx]
        dec_sentence = [self.dec_bos_token + s + self.dec_eos_token for s in sentence]
        dec_sentence_wlabel = [self.dec_bos_token + self.label_prompt[label] + self.dec_sep_token + s + self.dec_eos_token for s in sentence]
        
        enc_tokenized = self.enc_tokenizer(sentence, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_enc_tokenseq_length, return_tensors='pt')
        dec_tokenized = self.dec_tokenizer(dec_sentence, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_dec_tokenseq_length+2, return_tensors="pt")
        dec_tokenized_wlabel = self.dec_tokenizer(dec_sentence_wlabel, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_dec_tokenseq_length+4, return_tensors="pt")
        
        label = torch.tensor(label)
        label = torch.nn.functional.one_hot(label.to(torch.int64), self.num_classes)
        
        return {'enc_input_ids': enc_tokenized['input_ids'].squeeze(), 'enc_attention_mask': enc_tokenized['attention_mask'].squeeze(), 
                'dec_input_ids': dec_tokenized['input_ids'].squeeze(), 'dec_attention_mask': dec_tokenized['attention_mask'].squeeze(), 
                'dec_input_ids_wlabel': dec_tokenized_wlabel['input_ids'].squeeze(), 'dec_attention_mask_wlabel': dec_tokenized_wlabel['attention_mask'].squeeze(), 
                'labels': label}

def testing_yelploader():
    encoder_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", eos_token='<|endoftext|>', pad_token='<|pad|>')

    yelp_path = "/local1/pshah7/insnet/data/yelp/raw"

    train_dataset = YelpDataset(yelp_path, 'sentiment.train.1', 'sentiment.train.0', encoder_tokenizer, decoder_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return next(iter(train_dataloader))

if __name__ == "__main__":
    
    yelp_batch = testing_yelploader()
    
    from IPython import embed; embed(); exit()