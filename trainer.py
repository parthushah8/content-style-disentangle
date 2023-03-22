import torch
from torch.optim import AdamW

from transformers import RobertaTokenizer, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import YelpDataset
from models import RobertaEncoder, GPT2Decoder

from tqdm import tqdm
import wandb
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = 'saved_models'

def trainer(config):

    if config['encoder'] == 'roberta':
        encoder_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        encoder = RobertaEncoder.from_pretrained("roberta-base").to(device)

    if config['decoder'] == 'gpt2':
        decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", eos_token='<|endoftext|>', pad_token='<|pad|>')
        if config['add_sep_token']:
            sep_token = "<|sep_token|>"
            decoder_tokenizer.add_tokens(sep_token)
        decoder = GPT2Decoder(num_tokens=len(decoder_tokenizer)).to(device)

    if config['dataset'] == 'yelp':
        yelp_path = config['raw_data_dir']
        train_dataset = YelpDataset(yelp_path, 'sentiment.train.1', 'sentiment.train.0', encoder_tokenizer, decoder_tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, num_workers=config['num_workers'])
        val_dataset = YelpDataset(yelp_path, 'sentiment.dev.1', 'sentiment.dev.0', encoder_tokenizer, decoder_tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=True, num_workers=config['num_workers'])

    if config['optimizer'] == 'adam':
        classifier_opt = AdamW(lr = config['classifier_lr'], params=encoder.classification_head.parameters())
        enc_dec_opt = AdamW(lr = config['enc_dec_lr'], params=list(encoder.roberta.parameters()) + list(encoder.latent_projection.parameters()) + list(decoder.gpt2.parameters()))

    for epoch in range(config['total_epochs']+1):
        encoder.train()
        decoder.train()

        train_bce_loss, train_adv_loss, train_nll, train_nll_wlabel = 0.0, 0.0, 0.0, 0.0
        for i, batch in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
            # Send data from batch to device
            enc_input_ids, enc_attention_mask = batch['enc_input_ids'].to(device), batch['enc_attention_mask'].to(device)
            dec_input_ids, dec_attention_mask = batch['dec_input_ids'].to(device), batch['dec_attention_mask'].to(device)
            dec_input_ids_wlabel, dec_attention_mask_wlabel = batch['dec_input_ids_wlabel'].to(device), batch['dec_attention_mask_wlabel'].to(device)
            labels = batch['labels'].to(device)
            
            bce_loss = config['loss_alpha']['bce'] * bce_loss
            bce_loss.backward()
            classifier_opt.step()

            # Classifier gradient descent using just the bce loss
            classifier_opt.zero_grad()
            content_embedding, bce_loss, _ = encoder(
                input_ids = enc_input_ids,
                attention_mask = enc_attention_mask,
                labels = labels
            )
            bce_loss = config['loss_alpha']['bce'] * bce_loss
            bce_loss.backward()
            classifier_opt.step()

            # Encoder-Decoder gradient descent using the adverserial loss and the two nll losses
            enc_dec_opt.zero_grad()
            content_embedding, _, adv_loss = encoder(
                input_ids = enc_input_ids,
                attention_mask = enc_attention_mask,
                labels = labels
            )
            nll = decoder(
                content_embedding = content_embedding,
                input_ids = dec_input_ids,
                attention_mask = dec_attention_mask
            )
            nll_wlabel = decoder(
                content_embedding = content_embedding,
                input_ids = dec_input_ids_wlabel,
                attention_mask = dec_attention_mask_wlabel
            )
            adv_loss, nll, nll_wlabel = config['loss_alpha']['adv'] * adv_loss, config['loss_alpha']['nll'] * nll, config['loss_alpha']['nll_wlabel'] * nll_wlabel
            overall_loss = adv_loss + nll + nll_wlabel
            overall_loss.backward()            
            enc_dec_opt.step()

            # Logging Step
            train_bce_loss += (bce_loss.item())
            train_adv_loss += (adv_loss.item())
            train_nll += (nll.item())
            train_nll_wlabel += (nll_wlabel.item())

            if (i+1)%config['logging_interval_batch'] == 0:
                # Log the average training loss in the last config['logging_interval_batch'] batches
                train_bce_loss, train_adv_loss, train_nll, train_nll_wlabel = train_bce_loss/config['logging_interval_batch'], train_adv_loss/config['logging_interval_batch'], train_nll/config['logging_interval_batch'], train_nll_wlabel/config['logging_interval_batch']
                wandb.log({
                    "batches": epoch*len(train_loader)+i, 
                    "train_bce_loss": train_bce_loss, 
                    "train_adv_loss": train_adv_loss, 
                    "train_nll": train_nll, 
                    "train_nll_wlabel": train_nll_wlabel})
                # print("Batch {}: Train BCE Loss = {:.4f}, Train Adv Loss = {:.4f}, Train NLL Loss = {:.4f}, Train NLL with Label Loss = {:.4f}".format(i+1, train_bce_loss, train_adv_loss, train_nll, train_nll_wlabel))
                train_bce_loss, train_adv_loss, train_nll, train_nll_wlabel = 0.0, 0.0, 0.0, 0.0

        # Evaluate the model on the validation set
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_bce_loss, val_adv_loss, val_nll, val_nll_wlabel = 0.0, 0.0, 0.0, 0.0
            for i, batch in tqdm(enumerate(val_loader, 0), unit="batch", total=len(val_loader)):
                
                enc_input_ids, enc_attention_mask = batch['enc_input_ids'].to(device), batch['enc_attention_mask'].to(device)
                dec_input_ids, dec_attention_mask = batch['dec_input_ids'].to(device), batch['dec_attention_mask'].to(device)
                dec_input_ids_wlabel, dec_attention_mask_wlabel = batch['dec_input_ids_wlabel'].to(device), batch['dec_attention_mask_wlabel'].to(device)
                labels = batch['labels'].to(device)

                # Classifier gradient descent using just the bce loss
                content_embedding, bce_loss, adv_loss = encoder(
                    input_ids = enc_input_ids,
                    attention_mask = enc_attention_mask,
                    labels = labels
                )
                # Encoder-Decoder gradient descent using the adverserial loss and the two nll losses
                nll = decoder(
                    content_embedding = content_embedding,
                    input_ids = dec_input_ids,
                    attention_mask = dec_attention_mask
                )

                nll_wlabel = decoder(
                    content_embedding = content_embedding,
                    input_ids = dec_input_ids_wlabel,
                    attention_mask = dec_attention_mask_wlabel
                )

                # Scaling of losses
                bce_loss, adv_loss, nll, nll_wlabel = config['loss_alpha']['bce'] * bce_loss, config['loss_alpha']['adv'] * adv_loss, config['loss_alpha']['nll'] * nll, config['loss_alpha']['nll_wlabel'] * nll_wlabel

                # Update the validation loss
                val_bce_loss += (bce_loss.item())
                val_adv_loss += (adv_loss.item())
                val_nll += (nll.item())
                val_nll_wlabel += (nll_wlabel.item())

                # Add other validation metrics : BLEU

            # Compute the average validation loss
            val_bce_loss, val_adv_loss, val_nll, val_nll_wlabel = train_bce_loss/len(val_loader), train_adv_loss/len(val_loader), train_nll/len(val_loader), train_nll_wlabel/len(val_loader)

        # Log the epoch number and average validation loss
        # print("Epoch {}: Val BCE Loss = {:.4f}, Val Adv Loss = {:.4f}, Val NLL Loss = {:.4f}, Val NLL with Label Loss = {:.4f}".format(epoch+1, val_bce_loss, val_adv_loss, val_nll, val_nll_wlabel))
        wandb.log({
                "epoch": epoch, 
                "val_bce": val_bce_loss, 
                "val_adv": val_adv_loss, 
                "val_nll": val_nll, 
                "val_nll_wlabel": val_nll_wlabel
                })

        if (epoch+1)%config['save_interval_epoch'] == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'enc_dec_opt_state_dict': enc_dec_opt.state_dict(),
                'classifier_opt_state_dict': classifier_opt.state_dict(),
            }, f"saved_models/{config['exp_name']}/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    # start a new wandb run to track this script
    config = {
        'exp_name': 'roberta_gpt2_eq_alpha',
        # Architecture parameters
        'encoder' : 'roberta',
        'decoder' : 'gpt2',
        'add_sep_token' : False,
        # Data Loader parameters
        'dataset' : 'yelp',
        'raw_data_dir' : "/local1/pshah7/insnet/data/yelp/raw",
        'train_batch_size' : 256,
        'val_batch_size' : 256,
        'num_workers' : 6,
        # Optimizer Parameters
        'optimizer' : 'adam',
        'classifier_lr' : 1e-5,
        'enc_dec_lr' : 1e-5,
        # Training hyperparameters
        'training_mode' : 'all', # 'classifier-only', 'autoencoder-only', 'all'
        'total_epochs' : 60,
        'loss_alpha' : {
            'bce' : 1,
            'adv' : 1,
            'nll' : 1,
            'nll_wlabel' : 1,
        },
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