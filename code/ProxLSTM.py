import torch
import torch.nn as nn
import torch.autograd as ag
from torch.autograd import Variable
from scipy import optimize
from scipy.optimize import check_grad
import numpy

class ProximalLSTMCell(ag.Function):
    def __init__(self, lstm):	# feel free to add more input arguments as needed
        super(ProximalLSTMCell, self).__init__()
        self.lstm_cell = lstm   # use LSTMCell as blackbox


    def forward(self, input, pre_h, pre_c, prox_epsilon=1):
        '''need to be implemented'''
        print("input.shape: ", input.shape)
        with torch.enable_grad():
            self.G_t = torch.zeros(input.shape[0], self.lstm_cell.hidden_size, self.lstm_cell.input_size)
            self.h_t, self.s_t = self.lstm_cell(input,(pre_h, pre_c))
            for i in range(self.s_t.size(-1)):
                g_t = ag.grad(self.s_t[:,i], input, grad_outputs=torch.ones_like(self.s_t[:,0]), retain_graph=True)
                self.G_t[:,i,:] = g_t[0]
            self.s_t = self.s_t.unsqueeze(2)
            G_t_transpose = torch.transpose(self.G_t, 1, 2)            #G_t.permute(1,0)
            mul = torch.matmul(self.G_t, G_t_transpose)
            my_eye = torch.eye(mul.shape[-1])
            my_eye = my_eye.reshape((1, my_eye.shape[0], my_eye.shape[0]))
            my_eye = my_eye.repeat(input.shape[0], 1, 1)
            self.inverse = torch.inverse(my_eye + prox_epsilon*mul)
            self.c_t = torch.matmul(self.inverse, self.s_t)
            self.c_t = self.c_t.squeeze()

            return (self.h_t, self.c_t)


    def backward(self, grad_h, grad_c):
        '''need to be implemented'''
        # grad_input = grad_pre_c = grad_pre_h = None
        print("self.inverse.shape: {}, grad_c.shape: {}".format(self.inverse.shape, grad_c.unsqueeze(2).shape))
        a = torch.matmul(self.inverse,grad_c.unsqueeze(2))
        # a = a.squeeze()
        a_transpose = torch.transpose(self.G_t, 1, 2)
        print("a.shape: ", a.shape)
        print("self.c_t.shape: ", self.c_t.unsqueeze(2).shape)
        print("transpose shape: ", self.c_t.unsqueeze(2).permute(0, 2, 1).shape)
        # print(torch.matmul(a, self.c_t.unsqueeze(2).permute(0, 2, 1)))
        grad_g1 = torch.matmul(a, self.c_t.unsqueeze(2).permute(0, 2, 1))
        grad_g2 = torch.matmul(self.c_t.unsqueeze(2), a.permute(0, 2, 1))
        print("========= : ", (grad_g1 + grad_g2).shape)
        print("self.G_t.shape: ", self.G_t.shape)
        grad_g = -torch.matmul(grad_g1 + grad_g2, self.G_t)
        # grad_g = -torch.matmul((torch.matmul(a, self.c_t.unsqueeze(2).permute(0, 2, 1)) + torch.matmul(self.c_t, a.transpose())), self.G_t)
        
        print("grad_c.shape: ", grad_c.unsqueeze(2).shape)
        print("self.inverse.shape", self.inverse.shape)
        grad_s = torch.matmul(grad_c.unsqueeze(2).permute(0, 2, 1), self.inverse)
        grad_s = grad_s.squeeze()

        print("grad_g.shape:" , grad_g.shape)
        print("grad_h.shape:" , grad_h.shape)
        print("grad_s.shape:" , grad_s.shape)

        return grad_g, grad_h, grad_s
        

