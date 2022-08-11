from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#general settings
num_epochs = 5000
print_interval = 1000

#generator settings
input_size = 1

batch_size = 512

n_layers = 1
h_size = 128

#distribution settngs
seq_len = 20
data_mean_array =  np.random.randint(1,10,seq_len) # distribution array: input for sequence with timestemps
print('Input Timestep Array: ',data_mean_array)
data_stddev = 1



#learning settings
d_learning_rate = 1e-3
g_learning_rate = 3e-3

#discriminator settings
d_input_size = 1
d_hidden_size = 128
d_output_size = 1


d_minibatch_size = 512
g_minibatch_size = 512

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.out(self.rnn(x)[0])
        
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)
        self.out = nn.Linear(hidden_size, 1) #outputsize 2 für entscheidung
        self.out1 = nn.Sigmoid()

    def forward(self, x):
        return self.out1(self.out(self.rnn(x)[0]))





#RNN Generator
G = RNN(input_size, h_size, n_layers) #create a model
D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, num_layers=n_layers)

criterion = nn.BCELoss() # Binary cross entropy calculates the loss between input and target. Positive calss 1.0 / negative class 0.0
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate ) 
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate )

def train_D_on_actual():
    real_data_timestemp_array = []

    for t in range(seq_len):
        real_data_timestemp_array.append(Normal(data_mean_array[t],data_stddev).sample((1,d_minibatch_size, d_input_size)))

    #real_data_timestemp_array = np.asarray(torch.stack(real_data_timestemp_array))
    real_data_timestemp_array = torch.cat(real_data_timestemp_array)
    #print(real_data_timestemp_array.shape)
    real_decision = (D( real_data_timestemp_array ))
 
    real_error = criterion( real_decision, torch.ones(seq_len, d_minibatch_size, d_input_size))  # ones = true
    #print(real_error)
        #real_error_array.append(real_error)
    real_error.backward()
    
    return real_data_timestemp_array#, real_error_array
#Training function for generated data
'''
    This is how to train the discriminator to recognize fake data. The discriminator learns data produced by the generator should give 0.0 output.
    Note: It is not accepting any data — just that produced by the generator.
'''
def train_D_on_generated() :
    noise_array = []
    fake_data_array = []
    fake_decision_array = []
    #fake_error_array = []
    for t in range(seq_len):
        noise_array.append(torch.randn(1, d_minibatch_size, input_size)) # old
    
    fake_data_array = (G( torch.cat(noise_array) ))
    #print(fake_data.size())
    
    fake_decision_array = (D(fake_data_array ))
    
    fake_error = criterion( fake_decision_array, torch.zeros(seq_len, d_minibatch_size, d_input_size))  # zeros = fake
        #fake_error_array.append(fake_error)
    fake_error.backward()
    #print(fake_error)
    #return fake_decision, fake_error

#Generator trainer
def train_G(): #function gives back generated fake data
    noise_array = []
    fake_data_array = []
    fake_decision_array = []
    for t in range(seq_len):
        noise_array.append(torch.randn(1, d_minibatch_size, input_size)) # old
    
    fake_data_array = (G( torch.cat(noise_array) ))
    #print(fake_data.size()) 
    
    fake_decision_array = D( fake_data_array )
    #noise = torch.randn(seq_len, g_minibatch_size, input_size) #generating: noise = random value vektor (old)
    #fake_data = G(noise) #run noise through RNN -> from generator created samples
    #fake_decision = D( fake_data )
    
    error = criterion( fake_decision_array, torch.ones(seq_len, g_minibatch_size, input_size) )  #  size with g_minibatch_size
    error.backward()
    #print(fake_decision_array[t])

    return error.item(), fake_data_array

#print('Generated real data:')
print(train_D_on_actual())
#print(train_D_on_generated())
#print(train_G())
#print(hs)

#exit() #first exit without loss fkt
#Algorithm

losses = []
for epoch in range(num_epochs):

    D.zero_grad()
    
    train_D_on_actual()    
    train_D_on_generated()
    d_optimizer.step()
    
    G.zero_grad()
    loss,generated = train_G()
    g_optimizer.step()
    
    losses.append( loss )
    if( epoch % print_interval) == (print_interval-1) :
        print( "Epoch %6d. Loss %5.3f" % ( epoch+1, loss ) )
        
#print( "Training complete" )
#print(generated,loss)

data = []

for i in range(1000):
    noise = torch.randn(seq_len, d_minibatch_size, input_size) 
    G_out = G( noise )
for t in range(seq_len):
    data.append((G_out[t].detach().numpy()).flatten())
#print(noise)
print( "Generator output:" )
print(G_out)

data_test= np.asarray(data[seq_len-1]).flatten() #last element for plotting
#np.save('Generated_data.npy', data)
print('Shape generataed data:',np.shape(G_out))
#print(data)

sns.distplot(data_test)
#print(generated)
plt.show()


exit()


######################################################################################################################

