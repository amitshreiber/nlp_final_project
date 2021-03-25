from transformers import BertForSequenceClassification, BertConfig

class BERTClassifer():


  def __init__(self, args):


    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = args.CLASS_NUMBER
    self.model = BertForSequenceClassification(config)
    #model.parameters

   def embed_and_predict(token_songs_batch):


     input_ids = token_song.get('input_ids').to(self.device)
     attention_mask = token_song.get('attention_mask').to(self.device)


