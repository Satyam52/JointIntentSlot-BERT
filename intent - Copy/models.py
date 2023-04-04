from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
device = torch.device('cuda')


class IntentClassification(BertPreTrainedModel):
    def __init__(self, config, num_intent_labels, intent_labels_token):
        super().__init__(config)

        # store params
        classifier_dropout = config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob
        self.num_intent_labels = num_intent_labels
        self.config = config
        self.intent_desc = None
        self.intent_labels_input_ids = torch.tensor(
            list(map(lambda x: x['input_ids'], intent_labels_token)))
        self.intent_labels_attention_mask = torch.tensor(
            list(map(lambda x: x['attention_mask'], intent_labels_token)))
        self.intent_labels_token_type_ids = torch.tensor(
            list(map(lambda x: x['token_type_ids'], intent_labels_token)))
        self.linear_layer = nn.Linear(config.hidden_size, config.hidden_size)

        # create dot product matrix
        self.bert = BertModel(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    def init_intent_desc(self, device=torch.device('cpu')):
        _, pooled_out = self.bert(input_ids=self.intent_labels_input_ids.to(device), 
                            attention_mask=self.intent_labels_attention_mask.to(device), 
                            token_type_ids=self.intent_labels_token_type_ids.to(device), 
                            return_dict=False)
        self.intent_desc = torch.tensor(pooled_out, requires_grad=False)
            

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                intent_label_ids=None, output_attentions=None, output_hidden_states=None, eval=False):

        if not eval:
            self.init_intent_desc(device=device)
        
        # print(pooled_out.shape)
        
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]    # [Seq states]
        pooled_output = outputs[1]      # [CLS]
        pooled_output = self.linear_layer(pooled_output)
        
        # print(pooled_out.shape, pooled_output.shape)
        intent_logits = torch.matmul(pooled_output, torch.transpose(self.intent_desc,0,1))
        # print(intent_logits)
       
        total_loss = 0
      
        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss
        
        outputs = ((intent_logits),) + outputs[2:]
        outputs = (total_loss,) + outputs  # (loss), ((intent logits)), (hidden_states), (attentions)
        return outputs