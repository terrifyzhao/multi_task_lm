from transformers.models.roformer.modeling_roformer import SequenceClassifierOutput, RoFormerPreTrainedModel, \
    RoFormerModel
from transformers.models.bert.modeling_bert import SequenceClassifierOutput, BertModel, BertPreTrainedModel
import torch.nn as nn


class RoFormerForMultiTask(RoFormerPreTrainedModel):
    def __init__(self, config, num_labels_list):
        super().__init__(config)
        self.num_labels_list = num_labels_list
        self.roformer = RoFormerModel(config)
        self.classifier = nn.ModuleList(
            [nn.Linear(config.hidden_size, count) for count in self.num_labels_list])
        # self.classifier = RoFormerClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier[task_id](sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels_list[task_id]), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
