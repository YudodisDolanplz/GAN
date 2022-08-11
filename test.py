from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.out(self.rnn(x)[0])
        
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, 1) #outputsize 2 für entscheidung
        self.out1 = nn.Sigmoid()

    def forward(self, x):
        return self.out1(self.out(self.rnn(x)[0]))


#general settings
num_epochs = 5000
print_interval = 1000

#generator settings
input_size = 1
seq_len = 2
batch_size = 256

n_layers = 1
h_size = 256

#distribution settngs
data_mean = [0,5] 
data_stddev = 1

#more distrinbutions for sequence configuration
data_mean_2 = 2 
data_stddev_2 = 1

#learning settings
d_learning_rate = 1e-3
g_learning_rate = 3e-3

#discriminator settings
d_input_size = 1
d_hidden_size = 256
d_output_size = 1


d_minibatch_size = 180
g_minibatch_size = 180

for t in range(range(len(data_mean))):
    #RNN Generator
    G = RNN(seq_len, batch_size, input_size) #create a model
    D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, num_layers=n_layers)

    criterion = nn.BCELoss() # Binary cross entropy calculates the loss between input and target. Positive calss 1.0 / negative class 0.0
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate ) 
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate )

    def train_D_on_actual(mu,sigma) :
        dist = Normal(mu, sigma)
        real_data = dist.sample( (1, d_minibatch_size, d_input_size) )#.requires_grad_()
        #print(real_data)
        real_decision = D( real_data ) 
        #print(real_decision)
        #print(torch.ones(seq_len, 1, h_size))
        real_error = criterion( real_decision, torch.ones(1, d_minibatch_size, d_input_size))  # ones = true
        #print(real_error)
        real_error.backward()
        #print(real_error)

        #return real_decision, real_error
    #Training function for generated data
    '''
        This is how to train the discriminator to recognize fake data. The discriminator learns data produced by the generator should give 0.0 output.
        Note: It is not accepting any data — just that produced by the generator.
    '''
    def train_D_on_generated() :
        noise = torch.randn(1, d_minibatch_size, input_size)
        fake_data = G( noise )
        #print(fake_data.size()) 
        fake_decision = D( fake_data )
        fake_error = criterion( fake_decision, torch.zeros(1, d_minibatch_size, d_input_size))  # zeros = fake
        fake_error.backward()
        #print(fake_error)
        #return fake_decision, fake_error

    #Generator trainer
    def train_G(): #function gives back generated fake data
        noise = torch.randn(1, g_minibatch_size, input_size) #generating: noise = random value vektor
        fake_data = G(noise) #run noise through RNN -> from generator created samples
        fake_decision = D( fake_data )
        error = criterion( fake_decision, torch.ones(1, g_minibatch_size, input_size) )  #  size with g_minibatch_size
        error.backward()
        return error.item(), fake_data

    #print('Generated real data:')
    #print(train_D_on_actual(data_mean, data_stddev))
    #print(train_D_on_generated())
    #print(train_G())
    #print(hs)

    #exit() #first exit without loss fkt
    #Algorithm

    losses = []
    for epoch in range(num_epochs):

        D.zero_grad()
        
        train_D_on_actual(data_mean, data_stddev)    
        train_D_on_generated()
        d_optimizer.step()
        
        G.zero_grad()
        loss,generated = train_G()
        g_optimizer.step()
        
        losses.append( loss )
        if( epoch % print_interval) == (print_interval-1) :
            print( "Epoch %6d. Loss %5.3f" % ( epoch+1, loss ) )
            
    #print( "Training complete" )
    #print(generated)
    data = []

    for i in range(1000):
        noise = torch.randn(seq_len, d_minibatch_size, input_size) 
        G_out = G( noise )
        data.append((G_out.detach().numpy()).flatten())
    

data= np.asarray(data).flatten()
np.save('Generated_data.npy', data)
print(data.shape)
print(data)
sns.distplot(data)
#print(generated)
plt.show()


exit()

