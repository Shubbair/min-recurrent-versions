import torch
import torch.nn as nn
import torch.nn.functional as F

# vanilla GRU
class LSTM_Cell(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(LSTM_Cell,self).__init__()
        self.linear_f = nn.Linear(input_dim + hidden_dim , hidden_dim)
        self.linear_i = nn.Linear(input_dim + hidden_dim , hidden_dim)
        self.linear_c_tilda = nn.Linear(input_dim + hidden_dim , hidden_dim)
        self.linear_o = nn.Linear(input_dim + hidden_dim , hidden_dim)
        
    def forward(self,x_t,h_prev,c_prev):
        combined = torch.cat((x_t,h_prev),dim=1)
        
        f_t = torch.sigmoid(self.linear_f(combined))
        i_t = torch.sigmoid(self.linear_i(combined))   
        c_tilda_t = torch.sigmoid(self.linear_c_tilda(combined))
        o_t = torch.sigmoid(self.linear_o(combined))
        
        c_t = f_t * (c_prev - 1) + i_t * c_tilda_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t,c_t

# minLSTM
class minLSTM_Cell(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(minLSTM_Cell,self).__init__()
        
        self.linear_f = nn.Linear(input_dim,output_dim)
        self.linear_i = nn.Linear(input_dim,output_dim)
        self.linear_h = nn.Linear(input_dim,output_dim)
        
    def forward(self, x_t, h_prev, c_prev):
        f_t = torch.sigmoid(self.linear_f(x_t))
        i_t = torch.sigmoid(self.linear_i(x_t))
        
        h_tilda_t = self.linear_h(x_t)
        
        f_prime_t = f_t / (f_t + i_t)
        i_prime_t = i_t / (f_t + i_t)
        
        h_t = f_prime_t * h_prev + i_prime_t * h_tilda_t
        
        # Align with LSTM interface by also returning a cell state.
        # minLSTM has no separate cell state; we mirror the hidden state.
        c_t = h_t
        return h_t, c_t


# log-space minLSTM
def g(x : torch.Tensor)->torch.Tensor:
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

# minGRU_log
class log_minLSTM_Cell(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(log_minLSTM_Cell,self).__init__()
        
        self.linear_f = nn.Linear(input_dim,output_dim)
        self.linear_i = nn.Linear(input_dim,output_dim)
        self.linear_h = nn.Linear(input_dim,output_dim)
        
    def forward(self, x_t, h_prev, c_prev):
        f_t = torch.sigmoid(self.linear_f(x_t))
        i_t = torch.sigmoid(self.linear_i(x_t))
        
        h_tilda_t = g(self.linear_h(x_t))
        
        f_prime_t = f_t / (f_t + i_t)
        i_prime_t = i_t / (f_t + i_t)
        
        h_t = f_prime_t * h_prev + i_prime_t * h_tilda_t
        
        # Mirror hidden state as cell state for API compatibility
        c_t = h_t
        return h_t, c_t
    

# parallel 
def log_g(x : torch.Tensor)->torch.Tensor:
    return torch.where(x >= 0,(x + 0.5).log(),-F.softplus(-x))

def parallel_scan_log(log_coeffs : torch.Tensor, log_values : torch.Tensor)->torch.Tensor:
    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(
    log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]

# minGRU_log parallel
# log_space minLSTM
class parallel_log_minLSTM(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(parallel_log_minLSTM,self).__init__()
        
        self.linear_f = nn.Linear(input_dim,output_dim)
        self.linear_i = nn.Linear(input_dim,output_dim)
        self.linear_h = nn.Linear(input_dim,output_dim)
        
    def forward(self,x,h_0):
        diff = F.softplus(-self.linear_f(x)) / -F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = torch.log(h_0).unsqueeze(1)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(log_f,torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        return h
