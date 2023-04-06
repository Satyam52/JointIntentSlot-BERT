from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn


class JointIntentSlot(BertPreTrainedModel):
    def __init__(self, config, num_intent_labels, num_slot_labels, num_domain_labels):
        super().__init__(config)

        # store params
        classifier_dropout = config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.num_domain_labels = num_domain_labels
        self.config = config

        # define layers
        self.bert = BertModel(config)
        self.domain_dropout = nn.Dropout(classifier_dropout)  # Not sure if same dropout can be used
        self.intent_dropout = nn.Dropout(classifier_dropout)
        self.slot_dropout = nn.Dropout(classifier_dropout)
        self.domain_classifier = nn.Linear(config.hidden_size, num_domain_labels)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, domain_label_ids=None,
                intent_label_ids=None, slot_label_ids=None, output_attentions=None, output_hidden_states=None):

        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]  # [Seq states]
        pooled_output = outputs[1]  # [CLS]

        domain_pooled_output = self.domain_dropout(pooled_output)
        intent_pooled_output = self.intent_dropout(pooled_output)
        sequence_output = self.slot_dropout(sequence_output)
        domain_logits = self.domain_classifier(domain_pooled_output)
        intent_logits = self.intent_classifier(intent_pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        loss_fct = nn.CrossEntropyLoss()

        # 1. Domain Softmax
        if domain_label_ids is not None:
            domain_loss = loss_fct(domain_logits.view(-1, self.num_domain_labels), domain_label_ids.view(-1))
            total_loss += domain_loss

        # 2. Intent Softmax
        if intent_label_ids is not None:
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 3. Slot Softmax
        if slot_label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_label_ids.view(-1))
            total_loss += slot_loss

        outputs = ((domain_logits, intent_logits, slot_logits),) + outputs[2:]
        # (loss), ((domain_logits, intent logits, slot_logits)), (hidden_states), (attentions)
        outputs = (total_loss,) + outputs
        return outputs
