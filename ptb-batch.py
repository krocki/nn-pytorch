# -*- coding: utf-8 -*-
# author: kmrocki

import numpy as np
import argparse, sys
import datetime, time
import random

### parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fname', type=str, default = './logs/' + sys.argv[0] + '.dat', help='log filename')
parser.add_argument('--batchsize', type=int, default = 1, help='batch size')
parser.add_argument('--hidden', type=int, default = 64, help='hiddens')
parser.add_argument('--seqlength', type=int, default = 25, help='seqlength')
parser.add_argument('--timelimit', type=int, default = 100, help='time limit (s)')

opt = parser.parse_args()
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sys.argv[0], opt)
logname = opt.fname
B = opt.batchsize
S = opt.seqlength
T = opt.timelimit

start = time.time()
with open(logname, "a") as myfile:
    entry = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sys.argv[0], opt
    myfile.write("# " + str(entry))
    myfile.write("\n#  ITER\t\tTIME\t\tTRAIN LOSS\n")

# data I/O
data = open('./ptb/ptb.train.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = opt.hidden # size of hidden layer of neurons
seq_length = opt.seqlength # number of steps to unroll the RNN for
learning_rate = 1e-1
B = opt.batchsize

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size, B)) # encode in 1-of-k representation
    for b in range(0,B):
        xs[t][:,b][inputs[t][b]] = 1

    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]), axis=0) # probabilities for next chars
    #for b in range(0,B):
    loss += -np.log(ps[t][targets[t,0],0])/np.log(2) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    for b in range(0,B):
        dy[targets[t][b], b] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += np.expand_dims(np.mean(dy,axis=1), axis=1)
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += np.expand_dims(np.mean(dhraw,axis=1), axis=1)
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n = 0
p = np.random.randint(len(data)-1-S,size=(B)).tolist()
_inputs = [[0] * S] * B
_targets = [[0] * S] * B
inputs = np.zeros((S,B), dtype=int)
targets = np.zeros((S,B), dtype=int)
hprev = np.zeros((hidden_size,B))
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
start = time.time()

t = time.time()-start
last=start
while t < T:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  for b in range(0,B):
      if p[b]+seq_length+1 >= len(data) or n == 0:
        hprev[:,b] = np.zeros(hidden_size) # reset RNN memory
        p[b] = np.random.randint(len(data)-1-S)
        print p[b]

      inputs[:,b] = [char_to_ix[ch] for ch in data[p[b]:p[b]+seq_length]]
      targets[:,b] = [char_to_ix[ch] for ch in data[p[b]+1:p[b]+seq_length+1]]

   # print inputs
   #  [[41 20 27  0]
   #  [ 1 43 42  1]
   #  [26 38  1 45]
   #  [43 33 25 32]
   #  [44 21 37 27]]

  # sample from the model now and then
  if n % 1000 == 0:
    sample_ix = sample(np.expand_dims(hprev[:,0], axis=1), inputs[0], 1000)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + np.mean(loss) * 0.001
  interval = time.time() - last
  if n % 100 == 0:
    last = time.time()
    t = time.time()-start
    print(t, "Iteration", n, "Train loss:", smooth_loss)
    entry = '{:5}\t\t{:3f}\t{:3f}\n'.format(n, t, smooth_loss/seq_length)
    with open(logname, "a") as myfile:
         myfile.write(entry)
    print 'iter %d, %.4f BPC' % (n, smooth_loss / seq_length) # print progress

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  for b in range(0,B):
    p[b] += seq_length # move data pointer
  n += 1 # iteration counter 
