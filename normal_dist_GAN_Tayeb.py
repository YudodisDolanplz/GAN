from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Define the distribution that we will learn.
data_mean = 3.0
data_stddev = 0.4
Series_Length = 30

#Define the the generator network
g_input_size = 20    
g_hidden_size = 150  
g_output_size = Series_Length

#Define the discriminator network
'''
    discriminator performs one output from their input data
        - True (1.0) means the input matches desired distribution
        - False (0.0) means the input does not match the distribution
'''
d_input_size = Series_Length
d_hidden_size = 75   
d_output_size = 1

#Define how to send data into the process
d_minibatch_size = 15
g_minibatch_size = 10
num_epochs = 5000
print_interval = 1000

#Set the learning rates
d_learning_rate = 3e-3
g_learning_rate = 8e-3

'''
    Define two functions to provide a true sample and random noise. The true sample trains the discriminator, the random noise feeds the generator.
'''
def get_real_sampler(mu, sigma):
    dist = Normal( mu, sigma )
    return lambda m, n: dist.sample( (m, n) ).requires_grad_()

def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()  # Uniform-dist data into generator, _NOT_ Gaussian

actual_data = get_real_sampler( data_mean, data_stddev )
noise_data  = get_noise_sampler()

#print(a, b, c)

#The generator
'''
    The generator is trained to output means that match our desired distribution. This is a pretty simple 4 layer network, takes in noise and produces an output.
        Note: The last layer should be capable of producing values at least twice the given mean.
                Don’t use a sigmoid to produce a distribution with a mean of 100 — sigmoid’s range is 0.0 .. 1.0
'''

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.xfer = torch.nn.SELU()
    def forward(self, x):
        x = self.xfer( self.map1(x) )
        x = self.xfer( self.map2(x) )
        return self.xfer( self.map3( x ) )
'''
#RNN Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity=torch.relu):
        super(Generator, self).__init__()
        self.lay = nn.RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size) #layers (1)
        self.lay1 = nn.Linear(hidden_size, output_size=1)
    def forward(self, x):
        return self.lay1( self.lay(x) ) 
'''

#The Discriminator
'''
    This network is a classic multilayer perceptron — really nothing special at all. It returns true/false based on the learned function.
        Note: The last layer should restrict to 0..1 (opposite of the generator) This allows us more choice in loss functions.

'''

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.elu = torch.nn.ELU()
    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        return torch.sigmoid( self.map3(x) )

#Create the two networks
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
#G = Generator(input_size,)
D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)

#Setup the learning rules:
'''
    - a loss function, for the discriminator only. This uses binary cross entropy (BCE)
    - optimizers for both networks. This uses stochastic gradient descent (SGD)
'''
criterion = nn.BCELoss()
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate ) 
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate )

#Training function for real data
'''
    This is how to train the discriminator to recognize real data. The discriminator learns real data produces 1.0 output.
'''
def train_D_on_actual() :
    real_data = actual_data( d_minibatch_size, d_input_size )
    real_decision = D( real_data )
    print(real_decision)
    real_error = criterion( real_decision, torch.ones( d_minibatch_size, 1 ))  # ones = true
    real_error.backward()
    print(real_error)
#Training function for generated data
'''
    This is how to train the discriminator to recognize fake data. The discriminator learns data produced by the generator should give 0.0 output.
    Note: It is not accepting any data — just that produced by the generator.
'''
def train_D_on_generated() :
    noise = noise_data( d_minibatch_size, g_input_size )
    fake_data = G( noise ) 
    fake_decision = D( fake_data )
    #print(fake_decision)
    fake_error = criterion( fake_decision, torch.zeros( d_minibatch_size, 1 ))  # zeros = fake
    fake_error.backward()
    #print(fake_error)

#Training function for the generator
'''
    Assume that the generator produces perfect data (i.e. the discriminator returns 1.0).
    Then learn how to improve the output from the generator based on the discriminators actual output.

            *****This is the key piece of a GAN: pass the error through both networks, but only update the generators weights*****
'''
def train_G():
    noise = noise_data( g_minibatch_size, g_input_size )
    #print(noise)
    fake_data = G( noise ) #noise durch generator laufen lassen -> vom generator generierte samples
    fake_decision = D( fake_data )
    error = criterion( fake_decision, torch.ones( g_minibatch_size, 1 ) ) 
    error.backward()
    return error.item(), fake_data


#Algorithm
'''
The algo works like this:

Step 1 is plain old batch learning, if the rest of the code were removed you would have a network that can identify the desired distribution
    - train the discriminator just like you would train any network
    - use both true and false (generated) samples to learn

Step 2 is the GAN difference
    - train the generator to produce, but don’t compare the output to a good sample
    - feed the sample generated output through the disciminator to spot the fake
    - backpropagate the error through the discriminator and the generator

So let’s think about the possible cases (in all cases only the generators parameters are updated in the step 2)
Discrimator perfect, Generator Perfect Generator makes a sample which is identified as 1.0. Error is 0.0, no learning occurs
Discrimator perfect, Generator Rubbish Generator makes noise which is identified as 0.0. Error is 1.0, error is propagated and the generator learns
Discrimator rubbish, Generator Perfect Generator makes sample which is identified as 0.0. Error is 1.0, error is propagated the generator would not learn much because the error would be absorbed by the discriminator
Discrimator rubbish, Generator Rubbish Generator makes sample which is identified as 0.5. Error is 0.5, error is propagated the gradients in the discriminator and generator would mean the error is shared between both and learning occurs
This step can be slow — depending on the compute power available
'''
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
#print(generated)
data = []

for i in range(1000):
    noise = noise_data(d_minibatch_size, g_input_size) 
    G_out = G( noise )
    data.append((G_out.detach().numpy()).flatten())

data= np.asarray(data).flatten()
#print('-----data here-----')
#print(data)
sns.distplot(data)
plt.show()
print(train_D_on_actual())
exit()

#Displaying the results
'''
    After training everything we will generate some samples and draw them.
    It is easy enough to see that we have something like a nice normal distribution.
    This step is entirely optional but it is nice to see what we did actually worked.
'''
def draw( data ) :    
    plt.figure()
    d = data.tolist() if isinstance(data, torch.Tensor ) else data
    print(len(d),len(d[0]))
    plt.plot( d ) 
    plt.show()
d = torch.empty( generated.size(0), 53 ) 
for i in range( 0, d.size(0) ) :
    d[i] = torch.histc( generated[i], min=0, max=5, bins=53 )
draw( d.t() )