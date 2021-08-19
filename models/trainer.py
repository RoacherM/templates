import os
import torch
from tqdm import tqdm


class Trainer(object):
    
    def __init__(self, model, optim, loss, scheduler, evaluator, config=None):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.config = config
        self.initialize()
        self.history = {"train_acc": [], "train_loss": [], "valid_acc": [], "valid_loss": []}
        
    def initialize(self):
        if os.path.exists(self.config.model_path):
            self.model.load_state_dict(torch.load(self.config.model_path))
            print(f"parameters loaded from {self.config.model_path}")
        self.model.to(self.config.device)
        print("model initialized")

    def fit(self, train_data, valid_data):
        print("start training....")
        best_acc = 0.0
        batch_ix = 0

        for e in tqdm(range(self.config.epochs)):
            print("*"*50 + f"epoch:{e}" + "*"*50)
            ## prepare for train
            self.model.train()
            train_true, train_pred, train_loss = [], [], []
            for train in train_data:
                batch_ix += 1
                self.optim.zero_grad()
                ## 需要将数据和标签进行有效拆分，不要放一起，不然不好抽象
                inputs = torch.tensor(train[0], dtype=torch.long).to(self.config.device)
                labels = torch.tensor(train[1], dtype=torch.long).to(self.config.device)
                ## 前向计算过程
                logits = self.model(inputs)
                ## 计算损失函数
                losses = self.loss(logits, labels)
                
                train_true.append(labels.cpu().float())
                train_pred.append(logits.cpu().float())
                train_loss.append(losses.cpu().float())

                ## 每100轮评估一次
                if batch_ix % self.config.iters == 0:
                    ## prepare for valid
                    self.model.eval()
                    with torch.no_grad():
                        valid_true, valid_pred, valid_loss = [], [], []
                        for valid in valid_data:
                            inputs_ = torch.tensor(valid[0], dtype=torch.long).to(self.config.device)
                            labels_ = torch.tensor(valid[1], dtype=torch.long).to(self.config.device)

                            logits_ = self.model(inputs_)
                            ## 计算损失函数
                            losses_ = self.loss(logits_, labels_)

                            valid_true.append(labels_.cpu().float())
                            valid_pred.append(logits_.cpu().float())
                            valid_loss.append(losses_.cpu().float())     
                    ## 计算正确率
                    train_pred_label = torch.max(torch.cat(train_pred), 1)[1].numpy().astype("int")
                    train_true_label = torch.cat(train_true).numpy().astype("int")

                    valid_pred_label = torch.max(torch.cat(valid_pred), 1)[1].numpy().astype("int")
                    valid_true_label = torch.cat(valid_true).numpy().astype("int")

                    train_acc = self.evaluator(train_true_label, train_pred_label)["accuracy"]
                    valid_acc = self.evaluator(valid_true_label, valid_pred_label)["accuracy"]

                    train_loss_avg = sum(train_loss)/len(train_loss)
                    valid_loss_avg = sum(valid_loss)/len(valid_loss)

                    self.history["train_acc"].append(train_acc)
                    self.history["valid_acc"].append(valid_acc)
                    self.history["train_loss"].append(train_loss_avg.detach().numpy())
                    self.history["valid_loss"].append(valid_loss_avg.detach().numpy())

                    print(f"batch:{batch_ix}\n")
                    print(f"train_acc:{train_acc}  train_loss:{train_loss_avg}")
                    print(f"valid_acc:{valid_acc}  valid_loss:{valid_loss_avg}")
                    print("\n")

                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        torch.save(self.model.state_dict(), self.config.model_path)

                losses.backward()
                self.optim.step()
                self.scheduler.step()

    def predict(self, test_data, with_probs=False):
        ## 输入为包含一组str的list
        self.model.eval()
        with torch.no_grad():
            test_pred = []
            for test in test_data:
                inputs = torch.tensor(test[0], dtype=torch.long).to(self.config.device)
                labels = None
                
                logits = self.model(inputs)
                test_pred.append(logits.cpu().float())

        output = (torch.max(torch.cat(test_pred), 1)[1].numpy().astype("int"),)

        if with_probs:
            m = torch.nn.Softmax(dim=1)
            test_pred = [m(pred) for pred in test_pred]
            probs = torch.max(torch.cat(test_pred), 1)[0].numpy().astype("float")
            output = output + (probs,)
        return output