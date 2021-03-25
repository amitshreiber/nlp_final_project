from transformers import BertForSequenceClassification, BertConfig

class BERTClassifer():


  def __init__(self, args, device):


    # config = BertConfig.from_pretrained('bert-base-uncased')
    # config.num_labels = args.class_number
    # self.model = BertForSequenceClassification(config)

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


  def embed_and_predict(self,token_songs_batch, device):

     b_input_ids =token_songs_batch[0].to(device)
     b_input_mask =token_songs_batch[1].to(device)
     b_labels =token_songs_batch[2].to(device)
     b_labels =  b_labels.squeeze_()

     loss, logits = self.model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask,
                          labels=b_labels)
     return(logits)





