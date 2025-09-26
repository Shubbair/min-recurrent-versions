import torch
import torch.nn as nn
import torch.nn.functional as F

# vanilla GRU
class GRU_Cell(nn.Module):
    def __init__(self,input_dim : int,hidden_dim : int):
        super(GRU_Cell, self).__init__()
        
        self.linear_r = nn.Linear(input_dim + hidden_dim, hidden_dim) # reset gate 
 
        self.linear_z = nn.Linear(input_dim + hidden_dim, hidden_dim) # update gate 
                
        self.linear_h = nn.Linear(input_dim + hidden_dim, hidden_dim) # hidden candidate state
        
        # input_dim + hidden_dim , because it takes the combine (1,2)
        
    def forward(self,x_t : torch.Tensor, h_prev : torch.Tensor)->torch.Tensor:        

        combined = torch.cat((x_t,h_prev),dim=1)
        
        r_t = torch.sigmoid(self.linear_r(combined))
        
        z_t = torch.sigmoid(self.linear_z(combined))
        
        h_combine = torch.cat((x_t,r_t * h_prev),dim=1)
        h_candidate_t = torch.tanh(self.linear_h(h_combine))
        
        h_t = (1 - z_t) * h_prev + z_t * h_candidate_t
        return h_t
    
# minGRU
class minGRU_Cell(nn.Module):
    def __init__(self,input_dim : int,hidden_dim : int):
        super(minGRU_Cell, self).__init__()
        
        self.linear_z = nn.Linear(input_dim,hidden_dim)
        self.linear_h = nn.Linear(input_dim,hidden_dim)
        
    def forward(self,x_t : torch.Tensor, h_prev : torch.Tensor)->torch.Tensor:
        
        z_t = torch.sigmoid(self.linear_z(x_t))
        h_tilda_t = self.linear_h(x_t)
        h_t = (1 - z_t) * h_prev + z_t * h_tilda_t
        return h_t

# log-space minGRU
def g(x : torch.Tensor)->torch.Tensor:
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

# minGRU_log
class log_minGRU_Cell(nn.Module):
    def __init__(self,input_dim : int,hidden_dim : int):
        super(log_minGRU_Cell, self).__init__()
        
        self.linear_z = nn.Linear(input_dim,hidden_dim)
        self.linear_h = nn.Linear(input_dim,hidden_dim)
        
    def forward(self,x_t : torch.Tensor, h_prev : torch.Tensor)->torch.Tensor:
        
        z_t = torch.sigmoid(self.linear_z(x_t))
        h_tilda_t = g(self.linear_h(x_t))
        h_t = (1 - z_t) * h_prev + z_t * h_tilda_t
        return h_t
    
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
class parallel_log_minGRU(nn.Module):
    def __init__(self,input_dim : int,hidden_dim : int ):
        super(parallel_log_minGRU, self).__init__()
        
        self.linear_z = nn.Linear(input_dim,hidden_dim)
        self.linear_h = nn.Linear(input_dim,hidden_dim)    
    
    def forward(self,x : torch.Tensor, h_prev : torch.Tensor)->torch.Tensor:
        
        log_z = -F.softplus(-self.linear_z(x))
        log_coeffs = -F.softplus(self.linear_z(x))
        log_h = log_g(h_prev).unsqueeze(1)
        log_h_tilda = log_g(self.linear_h(x))
        h_t = parallel_scan_log(log_coeffs,torch.cat([log_h,log_z + log_h_tilda],dim=1))
        return h_t
