
from pytorch_transformers import BertModel

class BERT_net():


   def __init__(self):

       ## Load pretrained model/tokenizer
       self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)





