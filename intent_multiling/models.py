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
        dropout_rate=0.0,
        pooling="mean",
    ):
        super().__init__()
        layer_dim = layer_dim if layer_dim else input_dim
        self.pooling = pooling
        ic_head = []

        # Create the intermediate layers
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


class IntentClassification(RobertaPreTrainedModel):
    def __init__(self, config, num_intent_labels):
        super().__init__(config)

        # store params
        classifier_dropout = 0.1
        self.num_intent_labels = num_intent_labels
        self.config = config

        # print(config)

        # define layers
        self.roberta = XLMRobertaModel(config)
        self.intent_classifier_head = IntentClassificationHead(
            input_dim=config.hidden_size, num_intent_labels=self.num_intent_labels, layer_dim=config.hidden_size
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        intent_label_ids=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
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

        # print(dir(outputs), outputs)
        # exit()

        sequence_output = outputs[0]  #  last_hidden_state
        pooled_output = outputs[1]  # pooler_output

        intent_logits = self.intent_classifier_head(sequence_output, attention_mask)

        total_loss = 0
        # Intent Softmax
        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        outputs = ((intent_logits),) + outputs[2:]
        outputs = (total_loss,) + outputs  # (loss), ((intent logits)), (hidden_states), (attentions)
        return outputs
