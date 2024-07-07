import paddle
from paddle.nn import Conv2D, Linear, Embedding
from paddle import to_tensor
import paddle.nn.functional as F
import pandas as pd
import numpy as np

train_loader = pd.read_csv('gossipcop_v3-1_style_based_fake.csv')
test_loader = pd.read_csv('gossipcop_v3-1_style_based_fake.csv')
class RNN(paddle.nn.Layer):
    def __init__(self):
        super(RNN, self).__init__()
        self.dict_dim = vocab["<pad>"]
        self.emb_dim = 128
        self.hid_dim = 128
        self.class_dim = 2
        self.embedding = Embedding(
            self.dict_dim + 1, self.emb_dim,
            sparse=False)
        self._fc1 = Linear(self.emb_dim, self.hid_dim)
        self.lstm = paddle.nn.LSTM(self.hid_dim, self.hid_dim)
        self.fc2 = Linear(19200, self.class_dim)

    def forward(self, inputs):
        # [32, 150]
        emb = self.embedding(inputs)
        # [32, 150, 128]
        fc_1 = self._fc1(emb)#第一层
        # [32, 150, 128]
        x = self.lstm(fc_1)
        x = paddle.reshape(x[0], [0, -1])
        x = self.fc2(x)
        x = paddle.nn.functional.softmax(x)
        return x

rnn = RNN()
paddle.summary(rnn,(32,150),"int64")



def train(model):
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=0.002, parameters=model.parameters())
    
    steps = 0
    Iters, total_loss, total_acc = [], [], []
    
    for epoch in range(3):
        for batch_id, data in enumerate(train_loader):
            steps += 1
            sent = data[0]
            label = data[1]
            
            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)

            if batch_id % 50 == 0:
                Iters.append(steps)
                total_loss.append(loss.numpy()[0])
                total_acc.append(acc.numpy()[0])

                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            
            loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        
        for batch_id, data in enumerate(test_loader):
            
            sent = data[0]
            label = data[1]

            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)
            
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        
        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)

        print("[validation] accuracy: {}, loss: {}".format(avg_acc, avg_loss))
        
        model.train()

    paddle.save(model.state_dict(),"model_final.pdparams")
    
    draw_process("trainning loss","red",Iters,total_loss,"trainning loss")
    draw_process("trainning acc","green",Iters,total_acc,"trainning acc")
        
model = RNN()
train(model)
