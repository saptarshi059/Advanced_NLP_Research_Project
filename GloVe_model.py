from torchtext.data import Iterator, BucketIterator
from torchtext.data import TabularDataset
from torchtext.data import Field
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEXT = Field(sequential=True, tokenize=lambda x: x.split(), lower=True) #spacy's performance is really good but it takes some time to execute.
LABEL = Field(sequential=False, use_vocab=False) #set use_vocab = False when the data is already numerical.

datafields = [("id", None),("conversation",TEXT), ("category", LABEL)]

#If skip_header is set to False, then the headers also get processed!
trn = TabularDataset(path="train_custom.csv", format='csv', skip_header=True, fields=datafields)
tst = TabularDataset(path='test_custom.csv', format='csv', skip_header=True, fields=datafields)

#Creating the vocabulary using GloVe embeddings.
TEXT.build_vocab(trn, vectors="glove.42B.300d")

train_iter = BucketIterator(
 dataset =trn, # we pass in the datasets we want the iterator to draw data from
 batch_size =64,
 device=device,
 sort_key=lambda x: len(x.conversation), # the BucketIterator needs to be told what function it should use to group the data.
 sort_within_batch=False,
 repeat=False, # we pass repeat=False because we want to wrap this Iterator layer.
 shuffle =False, #Experiment with this to see if you're getting improved performance.
 train =True #Whether the dataset is a training set or not.
 )

test_iter = Iterator(tst, batch_size=64, device=device, sort=False, sort_within_batch=False, repeat=False, shuffle=False)

class BatchWrapper:
    #This takes care of the variable assignments.
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            '''
            We use "getattr" here because we want to generalize our code. This function is similar to "batch.conversation". 
            But then we would need to change this line for different functions. getattr returns the value of an attribute of an object.
            '''
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            
            if self.y_vars is not None: # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))
            yield (x, y)

    #This returns the number of batches.
    def __len__(self):
        return len(self.dl)

train_dl = BatchWrapper(train_iter, "conversation", ["category"]) #(iterator, independent_variable, dependent_variable)
test_dl = BatchWrapper(test_iter, "conversation", ["category"])

from torch.autograd import Variable
import torchtext

class ClassifierNet(torch.nn.Module):
    def __init__(self, glove, num_class):
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag.from_pretrained(glove.vectors)
        self.fc = torch.nn.Linear(glove.dim, num_class)
        
    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)

glove = torchtext.vocab.GloVe(name="42B",dim=300)

num_classes = 6
learning_rate = 0.01
num_epochs = 10

net = ClassifierNet(glove, num_classes)
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

print(net)

print("Starting Training...")
net.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    # Loop over all batches
    for x, y in train_dl:
        optimizer.zero_grad()  # zero the gradient buffer.
        
        conversation, category = Variable(x), Variable(y)
        
        #Transposing the training data.
        conversation = conversation.t()

        outputs = net(conversation.to(device))

        # Note: The true category tensor has to always be a 1D tensor of values (labels) for CrossEntropy!
        loss = criterion(outputs.to(device), category.squeeze().long().to(device)) 

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch} | Training Loss: {epoch_loss}')

# Test the Model
print("Evaluating the Model...")
net.eval()

test_preds = []
true = []

for x, y in tqdm(test_dl):
    conversation, category = Variable(x), Variable(y)
    conversation = conversation.t()
    preds = net(conversation.to(device))
    _, predicted = torch.max(preds.data, 1)
    test_preds.extend(predicted.tolist())
    true.extend(y.squeeze().long().tolist())

total = len(true)
correct = 0
for i in range(total):
    if test_preds[i] == true[i]:
        correct += 1

print(f'Accuracy of the network on the {total} test articles: {100 * correct / total} %')