from transformers import BertForSequenceClassification, BertConfig

class BERTClassifer():


  def __init__(self, args, device):



    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    self.model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels= args.class_number,  # The number of output labels
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    if device.type == 'cuda':
     self.model.cuda()





