
import numpy as np, pandas as pd
import joblib
import json
import sys 
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 


import torch
import torch.optim as optim
from torch.nn import GRU, LSTM, ReLU, Linear, Embedding, Module, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader 



MODEL_NAME = "text_class_base_rnn_pytorch"

model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"


COST_THRESHOLD = float('inf')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


def get_loss(model, device, data_loader, loss_function):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for data in data_loader:
            input, label = data[0].to(device), data[1].to(device)
            output = model(input.view(input.shape[0], -1))
            loss = loss_function(output, label)
            loss_total += loss.item()
    return loss_total / len(data_loader)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self,index):
        # Get one item from the dataset
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)
    



class Net(Module):
    def __init__(self, rnn_unit, V, T, K, D, M):
        super().__init__()
        self.rnn_unit = rnn_unit
        self.V = V
        self.T = T 
        self.K = K 
        self.D = D 
        self.M = M 

        rnn = self._get_rnn_unit()
        self.embedding_layer = Embedding(self.V + 1, self.D) 
        self.rnn_layer = rnn(input_size = self.D, hidden_size = self.M, bidirectional = True) #because GRU and LSTM have a bidirectional argument 

        self.linear1 = Linear(2 * self.M, 10)
        self.relu_layer = ReLU()
        self.output_layer = Linear(in_features = 10, out_features = self.K)
        

    def forward(self, x):
        x = self.embedding_layer(x)    
        x, _ = self.rnn_layer(x)    
        x, _ = torch.max(x, 1)
        x = self.linear1(x)  
        x = self.relu_layer(x)  
        x = self.output_layer(x)
        return x
     
    def _get_rnn_unit(self):
        if self.rnn_unit == 'gru':
            return GRU 
        elif self.rnn_unit == 'lstm':
            return LSTM 
        else: 
            raise ValueError(f"RNN unit {self.rnn_unit} is unrecognized. Must be either lstm or gru.")

    def get_num_parameters(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):                
                nn = nn*s
            pp += nn
        return pp            


class Classifier(): 
    
    def __init__(self, vocab_size, max_seq_len, num_target_classes, 
                 rnn_unit="gru", embedding_size=30, latent_dim=32, batch_size=32, **kwargs):
        '''
        rnn_unit: one of 'gru' or 'lstm'. Casing doesnt matter.
        V: vocabulary size
        T: length of sequences
        K: number of target classes       
        D: embedding size
        M: # of neurons in hidden layer  
        '''

        self.rnn_unit = rnn_unit.lower()     
        self.V = vocab_size
        self.T = max_seq_len
        self.K = num_target_classes
        self.D = embedding_size
        self.M = latent_dim
        self.batch_size = batch_size


        self.net = Net(rnn_unit = self.rnn_unit, V = self.V, T = self.T, K = self.K, 
                       D = self.D, M = self.M) 
        
        self.net.to(device)
        # print(self.net.get_num_parameters())
        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.net.parameters(), 
            )
        self.print_period = 5
        

    def fit(self, train_X, train_y, valid_X=None, valid_y=None, epochs=100, verbose=0):        
        
        train_X, train_y = torch.LongTensor(train_X), torch.LongTensor(train_y)
        train_dataset = CustomDataset(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=int(self.batch_size), shuffle=True)        
        
        if valid_X is not None and valid_y is not None:
            valid_X, valid_y = torch.LongTensor(valid_X), torch.LongTensor(valid_y)   
            valid_dataset = CustomDataset(valid_X, valid_y)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=int(self.batch_size),  shuffle=True)
        else:
            valid_loader = None

        losses = self._run_training(train_loader, valid_loader, epochs,
                           use_early_stopping=True, patience=5,
                           verbose=verbose)
        return losses
    
    
    def _run_training(self, train_loader, valid_loader, epochs,
                      use_early_stopping=True, patience=10, verbose=1):
        best_loss = 1e7
        losses = []
        min_epochs = 1
        for epoch in range(epochs):
            self.net.train()
            for _, data in enumerate(train_loader, 0):
                inputs,  labels = data[0].to(device), data[1].to(device)
                # print(inputs); sys.exit()
                # Feed Forward
                output = self.net(inputs)
                # Loss Calculation
                loss = self.criterion(output, labels)
                # Clear the gradient buffer (we don't want to accumulate gradients)
                self.optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Weight Update: w <-- w - lr * gradient
                self.optimizer.step()
                
            current_loss = loss.item()            
            
            if use_early_stopping:
                # Early stopping
                if valid_loader is not None:
                    current_loss = get_loss(self.net, device, valid_loader, self.criterion)
                losses.append({"epoch": epoch, "loss": current_loss})
                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience and epoch >= min_epochs:
                        if verbose == 1: print('Early stopping!')
                        return losses
                
            else:
                losses.append({"epoch": epoch, "loss": current_loss})
            # Show progress
            if verbose == 1:
                if epoch % self.print_period == 0 or epoch == epochs-1:
                    print(f'Epoch: {epoch+1}/{epochs}, loss: {np.round(loss.item(), 5)}')
        return losses   
        
    
    def predict(self, X):
        X = torch.LongTensor(X).to(device)
        preds = torch.softmax(self.net(X), dim=-1).detach().cpu().numpy()
        return preds
    

    def summary(self):
        self.model.summary()
        
    
    def evaluate(self, x_test, y_test):         
        """Evaluate the model and return the loss and metrics"""
        if self.net is not None:
            x_test, y_test = torch.LongTensor(x_test), torch.LongTensor(y_test)
            dataset = CustomDataset(x_test, y_test)
            data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
            current_loss = get_loss(self.net, device, data_loader, self.criterion)   
            return current_loss      


    def save(self, model_path): 
        model_params = {
            "rnn_unit": self.rnn_unit,
            "vocab_size": self.V,
            "max_seq_len": self.T,
            "num_target_classes": self.K,
            "embedding_size": self.D,
            "latent_dim": self.M,
            "batch_size": self.batch_size,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        torch.save(self.net.state_dict(), os.path.join(model_path, model_wts_fname))


    @classmethod
    def load(ml, model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        classifier = ml(**model_params)
        classifier.net.load_state_dict(torch.load( os.path.join(model_path, model_wts_fname)))        
        return classifier
    

def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path):     
    try: 
        model = Classifier.load(model_path)
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path): 
    with open( os.path.join(f_path, history_fname), mode='w') as f:
        f.write( json.dumps(history, indent=2) )



def get_data_based_model_params(train_X, train_y, valid_X, valid_y ): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''  
    V = max(train_X.max(), valid_X.max()) + 1
    T = train_X.shape[1]
    K = len(set(train_y).union(set(valid_y)))
    return {"vocab_size": V, "max_seq_len": T, "num_target_classes": K}

