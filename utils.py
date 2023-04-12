import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_collate(batch):
    padded_batch_dict = {}

    # Stacking the labels
    padded_batch_dict['labels'] = torch.stack([item['labels'] for item in batch], dim=0)

    # Encoded sequence
    max_len = max([len(item['enc_input_ids']) for item in batch])
    # Iterate over the batch items and copy each item to the corresponding position in the padded_batch
    padded_batch = torch.full((len(batch), max_len), 0, dtype=batch[0]['enc_input_ids'].dtype)
    for i, item in enumerate(batch):
        padded_batch[i, : item['enc_input_ids'].size(0)] = item['enc_input_ids']
    padded_batch_dict['enc_input_ids'] = padded_batch
    # Same thing for its attention mask
    padded_batch = torch.full((len(batch), max_len), 0, dtype=batch[0]['enc_attention_mask'].dtype)
    for i, item in enumerate(batch):
        padded_batch[i, : item['enc_attention_mask'].size(0)] = item['enc_attention_mask']
    padded_batch_dict['enc_attention_mask'] = padded_batch

    # Decoded sequence
    max_len = max([len(item['dec_input_ids']) for item in batch])
    # Iterate over the batch items and copy each item to the corresponding position in the padded_batch
    padded_batch = torch.full((len(batch), max_len), 0, dtype=batch[0]['dec_input_ids'].dtype)
    for i, item in enumerate(batch):
        padded_batch[i, : item['dec_input_ids'].size(0)] = item['dec_input_ids']
    padded_batch_dict['dec_input_ids'] = padded_batch
    # Same thing for its attention mask
    padded_batch = torch.full((len(batch), max_len), 0, dtype=batch[0]['dec_attention_mask'].dtype)
    for i, item in enumerate(batch):
        padded_batch[i, : item['dec_attention_mask'].size(0)] = item['dec_attention_mask']
    padded_batch_dict['dec_attention_mask'] = padded_batch

    # Decoded sequence with prompts
    max_len = max([len(item['dec_input_ids_wprompt']) for item in batch])
    # Iterate over the batch items and copy each item to the corresponding position in the padded_batch
    padded_batch = torch.full((len(batch), max_len), 0, dtype=batch[0]['dec_input_ids_wprompt'].dtype)
    for i, item in enumerate(batch):
        padded_batch[i, : item['dec_input_ids_wprompt'].size(0)] = item['dec_input_ids_wprompt']
    padded_batch_dict['dec_input_ids_wprompt'] = padded_batch
    # Same thing for its attention mask
    padded_batch = torch.full((len(batch), max_len), 0, dtype=batch[0]['dec_attention_mask_wprompt'].dtype)
    for i, item in enumerate(batch):
        padded_batch[i, : item['dec_attention_mask_wprompt'].size(0)] = item['dec_attention_mask_wprompt']
    padded_batch_dict['dec_attention_mask_wprompt'] = padded_batch

    return padded_batch_dict

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b