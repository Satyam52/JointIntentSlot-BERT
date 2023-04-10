from transformers import XLMRobertaModel, RobertaPreTrainedModel
import torch.nn as nn
import torch


class IntentClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim,
        num_intent_labels,
        layer_dim=None,
        num_layers=1,
        activation="gelu",
        dropout_rate=0.1,
        pooling="max",
    ):
        super().__init__()
        layer_dim = layer_dim if layer_dim else input_dim
        self.pooling = pooling
        ic_head = []

        # Create the hidden layers
        if num_layers > 0:
            for l in range(num_layers):
                ic_head.append(nn.Dropout(dropout_rate))
                ic_head.append(nn.Linear(input_dim, layer_dim))
                input_dim = layer_dim

                if activation == "gelu":
                    ic_head.append(nn.GELU())
                elif activation == "elu":
                    ic_head.append(nn.ELU())
                elif activation == "tanh":
                    ic_head.append(nn.Tanh())
                else:
                    raise NotImplementedError(f"Activation {activation} is not implemented")

        # Final layer, condensed to number of intent labels
        ic_head.append(nn.Dropout(dropout_rate))
        ic_head.append(nn.Linear(input_dim, num_intent_labels))

        self.ic_head = nn.Sequential(*ic_head)

    def forward(self, inp, attention_mask):
        if self.pooling == "first":
            # Get hidden states from first token in seq
            inp = inp[:, 0]
        elif self.pooling == "max":
            mask_expand = attention_mask.unsqueeze(-1).expand(inp.size()).float()
            inp[mask_expand == 0] = -1e9  # set padding to large negative
            inp = torch.max(inp, 1)[0]
        elif self.pooling == "mean":
            # see: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
            mask_expand = attention_mask.unsqueeze(-1).expand(inp.size()).float()
            inp = torch.sum(inp * mask_expand, 1) / torch.clamp(mask_expand.sum(1), min=1e-9)
        else:
            raise NotImplementedError(f"Pooling type {self.pooling} not implemented")

        return self.ic_head(inp)


class SlotFillingHead(nn.Module):
    def __init__(
        self,
        input_dim,
        num_slot_labels,
        layer_dim=None,
        num_layers=1,
        activation="gelu",
        dropout_rate=0.1,
    ):
        super().__init__()
        layer_dim = layer_dim if layer_dim else input_dim
        sf_head = []

        # Create the hidden layers
        if num_layers > 0:
            for l in range(num_layers):
                sf_head.append(nn.Dropout(dropout_rate))
                sf_head.append(nn.Linear(input_dim, layer_dim))
                input_dim = layer_dim

                if activation == "gelu":
                    sf_head.append(nn.GELU())
                elif activation == "elu":
                    sf_head.append(nn.ELU())
                elif activation == "tanh":
                    sf_head.append(nn.Tanh())
                else:
                    raise NotImplementedError(f"Activation {activation} is not implemented")

        # Final layer, condensed to number of intent labels
        sf_head.append(nn.Dropout(dropout_rate))
        sf_head.append(nn.Linear(input_dim, num_slot_labels))

        self.sf_head = nn.Sequential(*sf_head)

    def forward(self, inp):
        return self.sf_head(inp)


class JointIntentSlot(RobertaPreTrainedModel):
    def __init__(self, config, num_intent_labels, num_slot_labels, intent_labels_token, slot_labels_token):
        super().__init__(config)

        # store params
        classifier_dropout = config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.config = config
        self.intent_desc = None
        self.slot_desc = None
        self.intent_labels_token = intent_labels_token
        self.slot_labels_token = slot_labels_token

        # define layers
        self.roberta = XLMRobertaModel(config)
        self.intent_classifier = IntentClassificationHead(
            input_dim=config.hidden_size, num_intent_labels=self.num_intent_labels, layer_dim=256
        )
        self.slot_classifier = SlotFillingHead(
            input_dim=config.hidden_size, num_slot_labels=self.num_slot_labels, layer_dim=256
        )

        # Initialize weights and apply final processing
        self.post_init()

    def init_desc(self):
        if type == "intent":
            outputs = self.roberta(**self.intent_labels_token)
            self.intent_desc = outputs[1].detach()
        elif type == "slot":
            outputs = self.roberta(**self.slot_labels_token)
            self.slot_desc = outputs[1].detach()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        intent_label_ids=None,
        slot_label_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        eval=False,
    ):
        if not eval:
            self.init_desc()

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]  #  last_hidden_state
        pooled_output = outputs[1]  # pooler_output

        intent_logits = self.intent_classifier(sequence_output, attention_mask)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_label_ids.view(-1))
            total_loss += slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]
        outputs = (total_loss,) + outputs  # (loss), ((intent logits)), (hidden_states), (attentions)
        return outputs
