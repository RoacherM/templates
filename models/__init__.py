# import json
# import torch


# class Config(object):
#     def __init__(self):
#         self.max_len = 128
#         self.label2ids = json.load(open("./ocr/nlp_module/sentiment_analysis/common_analyzer/labels2id.json", "r", encoding="utf-8"))
#         self.id2labels = {val:key for key,val in self.label2ids.items()}
#         self.num_classes = len(self.label2ids)
        
        
#         self.dropout_ratio = 0.2
#         self.learning_rate = 5e-5 ##5e-6
        
#         self.pretrained_model = "albert" ## 支持bert/albert/robert
#         self.pretrained_model_path="./models/transformers/chinese_albert_small"
#         self.model_path = "./models/transformers/chinese_albert_small/model.bin"
        
#         self.batch_size = 32
#         self.epochs = 10
#         self.iters = 50
#         self.steps = 1000
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"


# class Config(object):
#     def __init__(self):
#         self.max_len = 128
#         self.label2ids = json.load(open("./ocr/nlp_module/sentiment_analysis/common_analyzer/labels2id.json", "r", encoding="utf-8"))
#         self.id2labels = {val:key for key,val in self.label2ids.items()}
#         self.num_classes = len(self.label2ids)
        
        
#         self.dropout_ratio = 0.2
#         self.learning_rate = 5e-5 ##5e-6
        
#         self.pretrained_model = "albert" ## 支持bert/albert/robert
#         self.pretrained_model_path="./models/transformers/chinese_albert_small"
#         self.model_path = "./models/transformers/chinese_albert_small/model.bin"
        
#         self.batch_size = 32
#         self.epochs = 10
#         self.iters = 50
#         self.steps = 1000
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"