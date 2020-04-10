
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import ProxLSTM as pro



class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, input_size):
		super(LSTMClassifier, self).__init__()
		

		self.output_size = output_size	# should be 9
		self.hidden_size = hidden_size  #the dimension of the LSTM output layer
		self.input_size = input_size	  # should be 12
		# self.normalize = F.normalize()
		self.conv = nn.Conv1d(in_channels= self.input_size, out_channels= 64, kernel_size= 3, stride= 1) # feel free to change out_channels, kernel_size, stride
		self.relu = nn.ReLU()
		self.lstm = nn.LSTM(64, hidden_size, batch_first = True)
		self.linear = nn.Linear(self.hidden_size, self.output_size)


		
	def forward(self, input, r, batch_size, mode='plain'):
		# do the forward pass
		# pay attention to the order of input dimension.
		# input now is of dimension: batch_size * sequence_length * input_size


		'''need to be implemented'''
		if mode == 'plain' :
				# chain up the layers
			# print ("input " + str(input.shape))
			normalized = F.normalize(input)
			# print ("normalized " + str(normalized.shape))
			embedding = self.conv(normalized.permute(0,2,1)).permute(0,2,1)
			# print ("embedding " + str(embedding.shape))
			lstm_input = self.relu(embedding)
			# print ("lstm_input " + str(lstm_input.shape))
			output, (h_n, c_n) = self.lstm(lstm_input)
			# print ("h_n " + str(h_n.squeeze().shape))
			decoded = self.linear(h_n.squeeze())
			return decoded
		"""
		if mode == 'AdvLSTM' :
			# chain up the layers
			# different from mode='plain', you need to add r to the forward pass
			# also make sure that the chain allows computing the gradient with respect to the input of LSTM

		if mode == 'ProxLSTM' :
			# chain up layers, but use ProximalLSTMCell here
		"""
		
# X = torch.tensor([[[1,2,3,4],
#               [5,6,7,8],
#               [9,10,11,12]],
# 			  [[21,22,23,24],
# 			  [25,26,27,28],
# 			  [29,30,31,32]]], dtype=torch.float32)
# print(X)
# Y = torch.flatten(X,0,1)
# print(Y)
# Z = Y.reshape(2,3,4)
# print(Z)
# ZZ = Z[:,-1]
# print(ZZ,ZZ.shape)
# model = nn.Linear(4, 9)
# p = model(Z[:,-1])
# print(p,p.shape)
# q = torch.argmax(p, dim = 1)
# print (q.shape)

# A = torch.tensor([2, 2, 8, 2, 4, 4, 4, 4, 4, 7, 7, 7, 8, 1, 2, 2, 4, 4, 4, 4, 7, 7, 7, 7,
#         7, 7, 7])
# print(A.view(-1,1))