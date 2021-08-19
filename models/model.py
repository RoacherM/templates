import torch
from transformers import BertTokenizer, RoFormerTokenizer
from transformers import BertModel, AlbertModel, RoFormerModel


class Tokenizer(object):
    """自定义tokenizer类"""
    def __init__(self, config=None):
        super(Tokenizer, self).__init__()
        if config.pretrained_model == "bert" or config.pretrained_model == "albert":
            self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_path)
        elif config.pretrained_model == "roformer":
            self.tokenizer = RoFormerTokenizer.from_pretrained(config.pretrained_model_path)
        else:
            raise Exception("model not supported yet!")


class BertTextClassifier(torch.nn.Module):
    """取last和first layer当作句向量表示"""
    def __init__(self, config=None):
        super().__init__()
        
        if config.pretrained_model == "bert": 
            self.bert = BertModel.from_pretrained(config.pretrained_model_path)
        elif config.pretrained_model == "albert":
            self.bert = AlbertModel.from_pretrained(config.pretrained_model_path)
        elif config.pretrained_model == "roformer":
            self.bert = RoFormerModel.from_pretrained(config.pretrained_model_path)
        else:
            raise Exception("model not supported yet!")

        self.activation = torch.nn.Tanh()
        self.dense = torch.nn.Linear(self.bert.config.hidden_size*2, config.num_classes)
        self.dropout = torch.nn.Dropout(config.dropout_ratio)
        
        torch.nn.init.xavier_normal_(self.dense.weight)
      
    def forward(self, inputs):
        # 分别取（除嵌入层之后的）第一层和最后一层首个token做句子分类
        layers = self.bert(input_ids=inputs, output_hidden_states=True, return_dict=True).hidden_states
        layer0 = self.activation(torch.mean(layers[1], dim=1))
        layer1 = self.activation(torch.mean(layers[-1], dim=1))
        logits = torch.cat((layer0, layer1), dim=1)
        logits = self.dropout(logits)
        logits = self.dense(logits)
        
        return logits