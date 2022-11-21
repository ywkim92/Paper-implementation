import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (BertPreTrainedModel, BertModel, 
                                                    BertForPreTrainingOutput, BertLMPredictionHead)
    
# BERT Sparse: Keyword-based Document Retrieval using BERT in Real time 
# reference: https://koreascience.kr/article/CFKO202030060897870.pdf
class BertForCustom(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer = False)
        self.predictions = BertLMPredictionHead(config)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        self.init_weights()

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        query_ids = None,
        query_mask = None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # labels=None,
        # next_sentence_label=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
    ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        
        # last_hidden_state: batch size * seq length * hidden size
        sequence_output,  = outputs[:1]
        
        # prediction_scores: batch size * seq length * vocab size
        # -> batch size * vocab size
        prediction_scores = self.predictions(sequence_output, ).transpose(1, 2)
        prediction_scores = self.gmp(prediction_scores).squeeze(-1)

        # query_ids: batch size * seq length
        # query_mask: batch size * seq length
        # torch.gather(prediction_scores, 1, query_ids[:, 1:]): batch size * (seq length - 1)
        # score: batch size
        score = (torch.gather(prediction_scores, 1, query_ids[:, 1:]) * query_mask[:, 1:]).sum(-1) / query_mask[:, 1:].sum(-1)
        return score