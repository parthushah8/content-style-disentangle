import torch
from torch.optim import AdamW

from transformers import RobertaTokenizer, GPT2Tokenizer
from torch.utils.data import DataLoader

from utils import pad_collate
from datasets import YelpDataset
from models import RobertaEncoder, GPT2Decoder

from tqdm import tqdm
import wandb
import os

# Device Configuration 
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Saved Model directory
checkpoint_dir = 'saved_models/autoencoders'

def trainer(config):

    # Initializing the Encoder and the Decoder model architecture
    if config['encoder'] == 'roberta':
        encoder_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        encoder = RobertaEncoder.from_pretrained("roberta-base").to(device)
    if config['decoder'] == 'gpt2':
        decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", eos_token='<|endoftext|>')
        decoder = GPT2Decoder(num_tokens=len(decoder_tokenizer)).to(device)

    # Creating the dataloader for the dataset to train the model over
    if config['dataset'] == 'yelp':
        yelp_path = config['raw_data_dir']
        train_dataset = YelpDataset(yelp_path, 'sentiment.train.1', 'sentiment.train.0', encoder_tokenizer, decoder_tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, collate_fn=pad_collate, num_workers=config['num_workers'])
        val_dataset = YelpDataset(yelp_path, 'sentiment.dev.1', 'sentiment.dev.0', encoder_tokenizer, decoder_tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=True, collate_fn=pad_collate, num_workers=config['num_workers'])

    # Optimizer used to train the architecure
    if config['optimizer'] == 'adam':
        enc_dec_opt = AdamW(lr = config['enc_dec_lr'], params=list(encoder.roberta.parameters()) + list(encoder.latent_projection.parameters()) + list(decoder.gpt2.parameters()))

    for epoch in range(config['total_epochs']+1):

        # Training Step
        encoder.train()
        decoder.train()
        train_nll = 0.0
        for i, batch in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
            # Send data from batch to device
            enc_input_ids, enc_attention_mask = batch['enc_input_ids'].to(device), batch['enc_attention_mask'].to(device)
            dec_input_ids, dec_attention_mask = batch['dec_input_ids'].to(device), batch['dec_attention_mask'].to(device)

            # Encoder-Decoder forward pass 
            enc_dec_opt.zero_grad()            
            content_embedding, _, _ = encoder(
                input_ids = enc_input_ids,
                attention_mask = enc_attention_mask
            )
            nll = decoder(
                content_embedding = content_embedding,
                input_ids = dec_input_ids,
                attention_mask = dec_attention_mask
            )

            # Backpropogation
            nll.backward()            
            enc_dec_opt.step()

            # Logging Step
            train_nll += (nll.item())

            if (i+1)%config['logging_interval_batch'] == 0:
                # Log the average training loss in the last config['logging_interval_batch'] batches
                train_nll = train_nll/config['logging_interval_batch']
                wandb.log({
                    "batches": epoch*len(train_loader)+i,
                    "train_nll": train_nll})
                print("Batch {}: Train NLL Loss = {:.4f}".format(i+1, train_nll))
                train_nll = 0.0

        # Evaluate the model on the validation set
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_nll = 0.0
            for i, batch in tqdm(enumerate(val_loader, 0), unit="batch", total=len(val_loader)):
                
                enc_input_ids, enc_attention_mask = batch['enc_input_ids'].to(device), batch['enc_attention_mask'].to(device)
                dec_input_ids, dec_attention_mask = batch['dec_input_ids'].to(device), batch['dec_attention_mask'].to(device)

                # Classifier gradient descent using just the bce loss
                content_embedding, _, _ = encoder(
                    input_ids = enc_input_ids,
                    attention_mask = enc_attention_mask                )
                # Encoder-Decoder gradient descent using the adverserial loss and the two nll losses
                nll = decoder(
                    content_embedding = content_embedding,
                    input_ids = dec_input_ids,
                    attention_mask = dec_attention_mask
                )
                # Update the validation loss
                val_nll += (nll.item())

            # Compute the average validation loss
            val_nll = val_nll/len(val_loader)

        # Log the epoch number and average validation loss
        print("Epoch {}: Val NLL Loss = {:.4f},".format(epoch+1, val_nll))
        wandb.log({
                "epoch": epoch, 
                "val_nll": val_nll
                })

        if (epoch+1)%config['save_interval_epoch'] == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'enc_dec_opt_state_dict': enc_dec_opt.state_dict(),
            }, f"{checkpoint_dir}/{config['exp_name']}/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    # start a new wandb run to track this script
    config = {
        'exp_name': 'autoencoder_v1',
        # Architecture parameters
        'encoder' : 'roberta',
        'decoder' : 'gpt2',
        # Data Loader parameters
        'dataset' : 'yelp',
        'raw_data_dir' : "/local1/pshah7/insnet/data/yelp/raw",
        'train_batch_size' : 4,
        'val_batch_size' : 4,
        'num_workers' : 6,
        # Optimizer Parameters
        'optimizer' : 'adam',
        'classifier_lr' : 1e-5,
        'enc_dec_lr' : 1e-5,
        # Training hyperparameters
        'total_epochs' : 60,
        'logging_interval_batch' : 200,
        'save_interval_epoch' : 5,
    }

    model_dir = f"{checkpoint_dir}/{config['exp_name']}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="style-transfer",
        entity='parthushah8',
        name=config['exp_name'],
        # track hyperparameters and run metadata
        config=config,
    )
    trainer(config)
    wandb.finish()