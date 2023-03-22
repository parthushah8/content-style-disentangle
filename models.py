import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead, RobertaConfig,
    Optional, Union, Tuple,
    SequenceClassifierOutput)
from transformers import GPT2Config, GPT2LMHeadModel
from utils import EntropyLoss

class RobertaEncoder(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.config = config
        self.content_size = config.hidden_size
        self.num_labels = 2

        self.bce_loss = nn.BCELoss()
        self.entropy_loss = EntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.latent_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(config.hidden_size, self.content_size),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(self.content_size, self.content_size),
            nn.LeakyReLU(0.1),
            nn.Linear(self.content_size, self.num_labels),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[Union[torch.Tensor, None]]:
        r"""
        labels : [batch_size, ], values in {0, 1}
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        content_encoding = self.latent_projection(outputs[0][:, 0, :])
        unnorm_logits = self.classification_head(content_encoding)

        if labels is not None:
            labels = labels.to(torch.float)
            flipped_labels = torch.where(labels == 1, torch.zeros_like(labels), torch.ones_like(labels))
            bce_loss = self.bce_loss(self.sigmoid(unnorm_logits), labels)
            adv_loss = self.bce_loss(self.sigmoid(unnorm_logits), flipped_labels)
            # from IPython import embed; embed(); exit()
        else:
            bce_loss, adv_loss = None, None

        return content_encoding, bce_loss, adv_loss
    
class GPT2Decoder(nn.Module): ## HF doc for PretrainModel

    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens

        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2.resize_token_embeddings(self.num_tokens)

    def forward(
        self,
        content_embedding: torch.LongTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
    ) -> Tuple[Union[torch.Tensor, None]]:
        r"""
        labels : [batch_size, ], values in {0, 1}
        """
        # Input Embedding tensor concatenating content embedding given by the encoder
        position_ids = attention_mask.cumsum(dim=-1)
        inputs_embeds_text = self.gpt2.transformer.wte(input_ids) + self.gpt2.transformer.wpe(position_ids)
        inputs_embeds = torch.cat((
            content_embedding.unsqueeze(dim=1), inputs_embeds_text
        ), dim=1)

        # Passing the Input embedding tensor through the decoder
        decoder_output = self.gpt2.forward(
            inputs_embeds = inputs_embeds
        )

        # Target Ids to calculate the NLL loss
        target_ids = input_ids[:, 1:]
        effective_mask = attention_mask[:, 1:].to(torch.float)

        # Calculate NLL Loss between the the Target Ids and the decoder logit output
        nll_loss = -decoder_output.logits[:, 2:, :].log_softmax(dim=-1).gather(
            dim=-1, index=target_ids.unsqueeze(dim=-1)).reshape_as(target_ids)
        nll_loss = (nll_loss * effective_mask).sum(dim=-1).mean(dim=-1)

        ## Doubt : Depending on which case I am in should I start picking from the fourth 
        ##         tokens in cases when I have some prompt to begin with

        return nll_loss
