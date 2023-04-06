from transformers import XLMRobertaModel, RobertaPreTrainedModel
import torch.nn as nn


class IntentClassification(RobertaPreTrainedModel):
    def __init__(self, config, num_intent_labels):
        super().__init__(config)

        # store params
        classifier_dropout = 0.1
        self.num_intent_labels = num_intent_labels
        self.config = config

        # define layers
        self.xlmr = XLMRobertaModel(config)
        self.dropout = nn.Dropout(classifier_dropout)
        self.activation = nn.GELU()
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels) 

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                intent_label_ids=None, output_attentions=None, output_hidden_states=None):

        outputs = self.xlmr(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]    # [Seq states]
        pooled_output = outputs[1]      # [CLS]

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        total_loss = 0
        # Intent Softmax
        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        outputs = ((intent_logits),) + outputs[2:]
        outputs = (total_loss,) + outputs  # (loss), ((intent logits)), (hidden_states), (attentions)
        return outputs
