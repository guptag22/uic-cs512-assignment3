import torch
import torch.nn as nn
import torch.autograd as ag
from scipy import optimize
from scipy.optimize import check_grad
import numpy

class ProximalLSTMCell(ag.Function):
    def __init__(self, lstm):	# feel free to add more input arguments as needed
        super(ProximalLSTMCell, self).__init__()
        self.lstm_cell = lstm   # use LSTMCell as blackbox


    def forward(self, input, pre_h, pre_c, r=0, prox_epsilon=1):

        '''need to be implemented'''
        with torch.enable_grad():
            self.h_t, self.s_t = self.lstm_cell(input,(pre_h, pre_c))
        print("s_t.shape: {}, input.shape: {}".format(self.s_t.shape, input.shape))
        G_t = ag.grad(outputs=self.s_t, inputs=input, grad_outputs=torch.ones_like(self.s_t), retain_graph=True, only_inputs=True, allow_unused=True)
        print("G_t: ", G_t)
        G_t = G_t[0]
        print("G_t.shape: ", G_t.shape)
        G_t_transpose = torch.transpose(G_t, 0, 1)            #G_t.permute(1,0)
        print("G_t_transpose.shape: ", G_t_transpose.shape)
        mul = torch.matmul(G_t, G_t_transpose)
        print("mul.shape: ", mul.shape)
        my_eye = torch.eye(mul.shape[0])
        # my_eye = my_eye.reshape((1, my_eye.shape[0], my_eye.shape[0]))
        # my_eye = my_eye.repeat(input.shape[0], 1, 1)
        print("my_eye.shape: ", my_eye.shape)
        inverse = torch.inverse(my_eye + prox_epsilon*mul)
        print("inverse.shape: ", inverse.shape)
        self.c_t = torch.matmul(inverse, self.s_t)
        print("self.c_t.shape: ", self.c_t.shape)
        return (self.h_t, self.c_t)

    # def get_gt(self, input):
    #     print("s_t.shape: ", self.s_t.shape)
    #     print("s_t: ", self.s_t.requires_grad)
    #     # TODO: double check grad_outputs argument
    #     G_t = ag.grad(outputs=self.s_t, inputs=input, grad_outputs=torch.ones_like(s_t), retain_graph=True, only_inputs=True, allow_unused=True)
    #     return G_t

    def backward(self, grad_h, grad_c):




        '''need to be implemented'''
