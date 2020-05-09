from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from torch.autograd import Variable
from collections import Counter
import pandas as pd
import numpy as np
import torch
import os

#For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Getting the path to the Data folder.
pwd = os.getcwd()
pwd = pwd.replace('Utils','Data')

def TFIDF(data):
    vectorizer_x = TfidfVectorizer(max_features=51)
    Data_TDM = vectorizer_x.fit_transform(data).toarray()
    return Data_TDM

train_df = pd.read_csv(pwd+'/train_custom.csv')
test_df = pd.read_csv(pwd+'/test_custom.csv')

#Majority Classifier Baseline
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)
dummy_clf.fit(train_df['conversation'], train_df['category'])
print(f"Majority Classification Baseline = {dummy_clf.score(test_df['conversation'],test_df['category']) * 100} %")

print("Creating the Feature Vectors...")
X_train, X_test = TFIDF(train_df['conversation']), TFIDF(test_df['conversation'])
Y_train, Y_test = train_df['category'], test_df['category']

class ClassifierNet(torch.nn.Module):
     def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_size,hidden_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = torch.nn.Linear(hidden_size, num_classes, bias=True)

     def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

input_size =  X_train.shape[1] #THE INPUT TO THE NN SHOULD ALWAYS BE THE DIMENSION OF THE FEATURE VECTORS!!
hidden_size = 100
num_classes = 6
learning_rate = 0.01
num_epochs = 10

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

net = ClassifierNet(input_size, hidden_size, num_classes)
net.to(device)

criterion = torch.nn.CrossEntropyLoss()  
criterion.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

print(net)

print("Starting Training...")
net.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()  # zero the gradient buffer    
    articles = Variable(torch.from_numpy(np.asarray(X_train)).float())
    labels = Variable(torch.from_numpy(np.asarray(Y_train)))
    outputs = net(articles.to(device))
    loss = criterion(outputs.to(device), labels.to(device))
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch} | Training Loss: {loss.item()} ')

# Test the Model
print("Evaluating the Model...")
net.eval()

correct = 0
total = len(test_df)

articles = Variable(torch.from_numpy(np.asarray(X_test)).float())
labels = Variable(torch.from_numpy(np.asarray(Y_test)))

outputs = net(articles.to(device))

_, predicted = torch.max(outputs.data, 1)

correct = (predicted.to(device) == labels.to(device)).sum()

print(f'Accuracy of the network on the {total} test articles: {100 * correct / total} %')

print("Confusion Matrix...")
print(pd.crosstab(pd.Series(labels.tolist()), pd.Series(predicted.tolist()), rownames=['True'], colnames=['Predicted'], margins=True))