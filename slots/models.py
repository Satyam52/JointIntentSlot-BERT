from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn


class SlotClassifier(BertPreTrainedModel):
    def __init__(self, config, num_slot_labels):
        super().__init__(config)

        # store params
        classifier_dropout = config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob
        self.num_slot_labels = num_slot_labels
        self.config = config

        # define layers
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(classifier_dropout)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels) 

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                slot_label_ids=None, output_attentions=None, output_hidden_states=None):

        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]    # [Seq states]
        pooled_output = outputs[1]      # [CLS]

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # Intent Softmax
        if slot_label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_label_ids.view(-1))
            total_loss += slot_loss

        outputs = ((slot_logits),) + outputs[2:]
        outputs = (total_loss,) + outputs  # (loss), ((intent logits)), (hidden_states), (attentions)
        return outputs
