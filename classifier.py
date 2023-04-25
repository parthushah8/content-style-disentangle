import torch
from torch.optim import AdamW

from transformers import RobertaTokenizer, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import YelpDataset
from models import RobertaEncoder, GPT2Decoder
from utils import pad_collate

from tqdm import tqdm
import wandb
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = 'saved_models'

def trainer(config):

    if config['encoder'] == 'roberta':
        encoder_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        encoder = RobertaEncoder.from_pretrained("roberta-base").to(device)

    if config['dataset'] == 'yelp':
        yelp_path = config['raw_data_dir']
        train_dataset = YelpDataset(yelp_path, 'sentiment.train.1', 'sentiment.train.0', encoder_tokenizer, None)
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, collate_fn=pad_collate, num_workers=config['num_workers'])
        val_dataset = YelpDataset(yelp_path, 'sentiment.dev.1', 'sentiment.dev.0', encoder_tokenizer, None)
        val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=True, collate_fn=pad_collate, num_workers=config['num_workers'])

    if config['optimizer'] == 'adam':
        classifier_opt = AdamW(lr = config['classifier_lr'], params=list(encoder.roberta.parameters()) + list(encoder.latent_projection.parameters()) + list(encoder.classification_head.parameters()))

    for epoch in range(config['total_epochs']+1):
        encoder.train()

        train_bce_loss = 0.0
        for i, batch in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
            # Send data from batch to device
            enc_input_ids, enc_attention_mask = batch['enc_input_ids'].to(device), batch['enc_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Classifier gradient descent using just the bce loss
            classifier_opt.zero_grad()
            _, bce_loss, _ = encoder(
                input_ids = enc_input_ids,
                attention_mask = enc_attention_mask,
                labels = labels
            )
            bce_loss = bce_loss
            bce_loss.backward()
            classifier_opt.step()

            # Logging Step
            train_bce_loss += bce_loss.item()

            if (i+1)%config['logging_interval_batch'] == 0:
                # Log the average training loss in the last config['logging_interval_batch'] batches
                train_bce_loss = train_bce_loss/config['logging_interval_batch']
                # wandb.log({
                #     "batches": epoch*len(train_loader)+i, 
                #     "train_bce_loss": train_bce_loss,
                #     })
                print("Batch {}: Train BCE Loss = {:.4f}".format(i+1, train_bce_loss))
                train_bce_loss = 0.0

        # Evaluate the model on the validation set
        encoder.eval()
        with torch.no_grad():
            val_bce_loss = 0.0
            for i, batch in tqdm(enumerate(val_loader, 0), unit="batch", total=len(val_loader)):
                
                enc_input_ids, enc_attention_mask = batch['enc_input_ids'].to(device), batch['enc_attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Classifier gradient descent using just the bce loss
                _, bce_loss, _ = encoder(
                    input_ids = enc_input_ids,
                    attention_mask = enc_attention_mask,
                    labels = labels
                )

                # Update the validation loss
                val_bce_loss += (bce_loss.item())

            # Compute the average validation loss
            val_bce_loss = val_bce_loss/len(val_loader)

        # Log the epoch number and average validation loss
        print("Epoch {}: Val BCE Loss = {:.4f}".format(epoch+1, val_bce_loss))
        # wandb.log({
        #         "epoch": epoch, 
        #         "val_bce": val_bce_loss,
        #         })

        if (epoch+1)%config['save_interval_epoch'] == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'classifier_opt_state_dict': classifier_opt.state_dict(),
            }, f"saved_models/{config['exp_name']}/model_epoch_{epoch}.pt")

def predict(config, input_texts):

    # Initializing the Encoder and the Decoder model architecture
    if config['encoder'] == 'roberta':
        encoder_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        encoder = RobertaEncoder.from_pretrained("roberta-base").to(device)

    # Load the saved model using the weights saved
    checkpoint = torch.load(f"{checkpoint_dir}/{config['exp_name']}/model_epoch_{config['last_epoch']}.pt")
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    # Get the content Embedding
    enc_tokenized = encoder_tokenizer.batch_encode_plus(input_texts, add_special_tokens=True, padding=True, return_tensors='pt').to(device)
    _, pred_probs = encoder(
        input_ids = enc_tokenized['input_ids'].squeeze(),
        attention_mask = enc_tokenized['attention_mask'].squeeze()
    )

    print(pred_probs)

    for i in range(len(input_texts)):
        print(f'Input Text : {input_texts[i]}')
        print(f'Output Text : {pred_probs[i]}')

    return

if __name__ == "__main__":
    # start a new wandb run to track this script
    # config = {
    #     'exp_name': 'roberta_classifier',
    #     # Architecture parameters
    #     'encoder' : 'roberta',
    #     'decoder' : 'gpt2',
    #     'add_sep_token' : False,
    #     # Data Loader parameters
    #     'dataset' : 'yelp',
    #     'raw_data_dir' : "/local1/pshah7/insnet/data/yelp/raw",
    #     'train_batch_size' : 512,
    #     'val_batch_size' : 512,
    #     'num_workers' : 6,
    #     # Optimizer Parameters
    #     'optimizer' : 'adam',
    #     'classifier_lr' : 1e-5,
    #     'enc_dec_lr' : 1e-5,
    #     # Training hyperparameters
    #     'training_mode' : 'autoencoder', # 'classifier', 'autoencoder', 'all'
    #     'total_epochs' : 60,
    #     'loss_alpha' : {
    #         'bce' : 1,
    #         'adv' : 1,
    #         'nll' : 1,
    #         'nll_wlabel' : 1,
    #     },
    #     'logging_interval_batch' : 200,
    #     'save_interval_epoch' : 5,
    # }

    # model_dir = f"{checkpoint_dir}/{config['exp_name']}"
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="style-transfer",
    #     entity='parthushah8',
    #     name=config['exp_name'],
    #     # track hyperparameters and run metadata
    #     config=config,
    # )
    # trainer(config)
    # wandb.finish()

    config = {
        'exp_name': 'roberta_classifier',
        # Architecture parameters
        'encoder' : 'roberta',
        'last_epoch' : 59,
    }

    model_dir = f"{checkpoint_dir}/{config['exp_name']}"
    if not os.path.exists(model_dir):
        print(f"Error: {config['exp_name']} No such Model Directory exists")

    input_texts = ["my goodness it was so gross .", "the house chow fun was also superb ."]

    predict(config, input_texts)