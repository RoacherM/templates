# from .config import config
from .models.trainer import Trainer
from .models.evaluator import evaluate ## 后期可以封装成类，便于参数化选择
from .models.model import BertTextClassifier, Tokenizer
from .models.losses import compute_ce_loss, compute_ce_loss_with_rdrop ## 后期可以封装成类，便于参数化选择

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, AdamW ## 训练时使用


###############预处理部分#######################
class DataSetIterater(Dataset):
    def __init__(self, texts, label):
        self.texts = texts
        self.label = label
        
    def __getitem__(self, item):
        return self.texts[item], self.label[item]
    
    def __len__(self):
        return len(self.texts)
    

###############文本分类器pipeline#################
class Classifier(object):
    
    def __init__(self, config=None):
        self.config = config
        self.tokenizer = Tokenizer(config).tokenizer

        ###############模型初始化#######################
        self.model = BertTextClassifier(config=config)
        ## 只有训练时使用
        self.optim = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_cosine_schedule_with_warmup(self.optim, num_warmup_steps=config.iters, 
                                                    num_training_steps=config.steps \
                                                    if config.steps is not None else 5*config.iters)

        self.trainer = Trainer(model=self.model, optim=self.optim, loss=compute_ce_loss, scheduler=self.scheduler, \
                        evaluator=evaluate, config=config)
        
    def predict(self, sentences):
        token_input = (sentences, [0]*len(sentences))
        token_batch = self.Loader(dataset=token_input, batch_size=1, shuffle=False)
        token_label, token_logit = self.trainer.predict(token_batch, with_probs=True)

        outs = []
        for sent, label, logit in zip(sentences, token_label, token_logit):
            outs.append((sent, self.config.id2labels[label], logit))
        return outs


    def fit(self, traindata, validdata):
        trainset = self.Loader(traindata, batch_size=self.config.batch_size, shuffle=True)
        validset = self.Loader(validdata, batch_size=self.config.batch_size, shuffle=True)

        ## 测试程序
        self.trainer.fit(train_data=trainset, valid_data=validset)

        return self.trainer.history
        
        
    def Loader(self, dataset, batch_size, shuffle=True):

        def collate_fn(batch_data, tokenizer=self.tokenizer, max_len=self.config.max_len):
            batch_tokens, batch_labels = [], []

            for text, label in batch_data:
                tokens = tokenizer.encode_plus(
                    text=text, 
                    add_special_tokens=True, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=max_len,
                    return_attention_mask=False,
                )
                ## 当使用rdrop的时候，每个样本repeat一次
        #         for i in range(2):
                batch_tokens.append(tokens["input_ids"])
                batch_labels.append(label)

            return batch_tokens, batch_labels

        data = DataSetIterater(dataset[0], dataset[1])
        gens = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return gens


# ###############模型初始化#######################
# model = BertTextClassifier(config=config)
# ## 只有训练时使用
# optim = AdamW(model.parameters(), lr=config.learning_rate)
# scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=config.iters, 
#                                             num_training_steps=config.steps \
#                                             if config.steps is not None else 5*config.iters)

# Model = Trainer(model=model, optim=optim, loss=compute_ce_loss, scheduler=scheduler, \
#                 evaluator=evaluate, config=config)

# ###############模型接口#######################
# def predict(sentences):
#     token_input = (sentences, [0]*len(sentences))
#     token_batch = Loader(dataset=token_input, batch_size=1, shuffle=False)
#     token_label, token_logit = Model.predict(token_batch, with_probs=True)

#     outs = []
#     for sent, label, logit in zip(sentences, token_label, token_logit):
#         outs.append((sent, config.id2labels[label], logit))
#     return outs


# def fit(traindata, validdata):
#     trainset = Loader(traindata, batch_size=config.batch_size, shuffle=True)
#     validset = Loader(validdata, batch_size=config.batch_size, shuffle=True)

#     ## 测试程序
#     Model.fit(train_data=trainset, valid_data=validset)

#     return Model.history
    
