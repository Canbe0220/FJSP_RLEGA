import copy
import torch
import math
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
from torch.distributions import Categorical
from mlp import MLPCritic, MLPActor

class MultiHeadExternalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0 
        self.coef = 8
        self.trans_dims = nn.Linear(dim, dim * self.coef)      
        self.k = dim // self.num_heads  
        self.num_heads = self.num_heads * self.coef
        self.linear_0 = nn.Linear(self.k, self.k)
        self.linear_1 = nn.Linear(self.k, self.k)

        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_parameters()  

    def reset_parameters(self):
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(self.trans_dims.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_0.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_1.weight, gain=gain)   
        nn.init.xavier_normal_(self.proj.weight, gain=gain)     

    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
        attn = self.linear_0(x)
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-10 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0,2,1,3).reshape(B, N, -1)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class GAT(nn.Module):
    
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0.):
        
        super(GAT, self).__init__()
        self.ope_dim = in_dim[0]
        self.mac_dim = in_dim[1]
        self.out_dim = out_dim
        self.nega_slope = 0.2

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.ope_w = nn.Linear(self.ope_dim, self.out_dim, bias=False)
        self.mac_w = nn.Linear(self.mac_dim, self.out_dim, bias=False)
        
        self.ope_alpha = nn.Parameter(torch.rand(size=(1, 1, self.out_dim), dtype=torch.float))
        self.mac_alpha = nn.Parameter(torch.rand(size=(1, 1, self.out_dim), dtype=torch.float))    
        
        self.leaky_relu = nn.LeakyReLU(self.nega_slope)
        self.activate = torch.tanh
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.ope_w.weight, gain=gain)
        nn.init.xavier_normal_(self.mac_w.weight, gain=gain)

        nn.init.xavier_normal_(self.ope_alpha, gain=gain)
        nn.init.xavier_normal_(self.mac_alpha, gain=gain)
        
    def forward(self, curr_proc_batch, batch_idxes, feats):

        feat_ope = self.feat_drop(feats[0])
        feat_mac = self.feat_drop(feats[1])
            
        h_ope = self.ope_w(feat_ope)
        h_mac = self.mac_w(feat_mac)
        
        # attention coefficients
        attn_ope = (h_ope * self.ope_alpha).sum(dim=-1).unsqueeze(-1)
        attn_mac = (h_mac * self.mac_alpha).sum(dim=-1).unsqueeze(-1)

        attn_ope = curr_proc_batch[batch_idxes].unsqueeze(-1) * (attn_ope.unsqueeze(-2) + attn_mac.unsqueeze(-3)) 
        e_ijk = self.leaky_relu(attn_ope)

        e_ijk = e_ijk.squeeze(-1)
        mask_ijk = (curr_proc_batch[batch_idxes]==1)
        e_ijk = e_ijk.masked_fill(~mask_ijk, float('-9e10'))
        alpha_ijk = F.softmax(e_ijk, dim=-2)
        alpha_ijk = alpha_ijk * mask_ijk.float()

        out_ope = torch.sum(h_ope.unsqueeze(-2) * alpha_ijk.unsqueeze(-1), dim=-3)
        out_mac = h_mac

        return out_ope + out_mac


# ablation study
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()

        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))

            return self.linears[self.num_layers - 1](h)


class DecisionMaking(nn.Module):
    def __init__(self, model_paras):
        super(DecisionMaking, self).__init__()

        self.device = model_paras["device"]
        self.in_ope_dim = model_paras["in_ope_dim"] 
        self.in_mac_dim = model_paras["in_mac_dim"]  
        self.out_ope_dim = model_paras["out_ope_dim"]
        self.out_mac_dim = model_paras["out_mac_dim"]
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        self.actor_in_dim = model_paras["actor_in_dim"]
        self.critic_in_dim = model_paras["critic_in_dim"]
        self.actor_layer = self.critic_layer = model_paras["policy_layer"]
        self.actor_hidden_dim = self.critic_hidden_dim = model_paras["policy_hidden_dim"] 
        self.actor_out_dim = self.critic_out_dim = model_paras["policy_out_dim"] 
        
        # EA-GA     
        self.get_opes = MultiHeadExternalAttention(self.in_ope_dim, self.num_heads)
        self.get_macs = GAT((self.in_ope_dim, self.in_mac_dim), self.out_mac_dim, self.dropout, self.dropout)
        

        self.actor = MLPActor(self.actor_layer, self.actor_in_dim, self.actor_hidden_dim, self.actor_out_dim).to(self.device)
        self.critic = MLPCritic(self.critic_layer, self.critic_in_dim, self.critic_hidden_dim, self.critic_out_dim).to(self.device)


    def act_prob(self, state, memory, flag_train=True):

        '''
        probability distribution
        '''

        curr_proc_adj = state.curr_proc_batch
        batch_idxes = state.batch_idxes
        raw_opes = state.feat_opes_batch[batch_idxes]
        raw_macs = state.feat_macs_batch[batch_idxes]
        
        # Normalize
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True)
        norm_opes = (raw_opes - mean_opes) / (std_opes + 1e-8)

        mean_macs = torch.mean(raw_macs, dim=-2, keepdim=True)
        std_macs = torch.std(raw_macs, dim=-2, keepdim=True)
        norm_macs = (raw_macs - mean_macs) / (std_macs + 1e-8)

        
        h_opes = self.get_opes(norm_opes)
        h_macs = self.get_macs(curr_proc_adj, batch_idxes, (norm_opes, norm_macs))   
     
        h_opes_pooled = h_opes.mean(dim=-2)
        h_macs_pooled = h_macs.mean(dim=-2)

        # expand and concatenate
        h_opes_expand = h_opes.unsqueeze(-2).expand(-1, -1, curr_proc_adj.size(-1), -1)
        h_macs_expand = h_macs.unsqueeze(-3).expand(-1, curr_proc_adj.size(-2), -1, -1)
        h_opes_pooled_expand = h_opes_pooled[:, None, None, :].expand_as(h_opes_expand)
        h_macs_pooled_expand = h_macs_pooled[:, None, None, :].expand_as(h_macs_expand)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)
        candidate_opes = ~(state.mask_job_procing_batch[batch_idxes] + state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_opes_expand[..., 0])
        idle_macs = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_opes_expand[..., 0])
        mask_proc = (curr_proc_adj[batch_idxes] == 1) & candidate_opes & idle_macs

        # actor MLP
        h_actions = torch.cat((h_opes_expand, h_macs_expand, h_opes_pooled_expand, h_macs_pooled_expand), dim=-1).transpose(1, 2)
        mask = mask_proc.transpose(1, 2).flatten(1)

        #priority probability
        prob = self.actor(h_actions).flatten(1)
        prob[~mask] = float('-inf')
        action_probs = F.softmax(prob, dim=1)           

        if flag_train:
            # using sample strategy during training
            dist = Categorical(action_probs)
            action_indexes = dist.sample()

            # Store memory data during training
            memory.logprobs.append(dist.log_prob(action_indexes))
            memory.action_indexes.append(action_indexes)
            memory.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memory.curr_proc_adj.append(copy.deepcopy(curr_proc_adj))
            memory.norm_opes.append(copy.deepcopy(norm_opes))
            memory.norm_macs.append(copy.deepcopy(norm_macs))
            memory.mask_proc.append(copy.deepcopy(mask_proc))
            
        else:
            # using greedy strategy during validating and testing
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / curr_proc_adj.size(-2)).long()
        jobs = (action_indexes % curr_proc_adj.size(-2)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]         

        return torch.stack((opes, mas, jobs), dim=1).t()


    def evaluate(self, curr_proc_adj, norm_opes, norm_macs, mask_proc, action_indexes):
        batch_idxes = torch.arange(0, curr_proc_adj.size(-3)).long()
        features = (norm_opes, norm_macs)

        
        h_opes = self.get_opes(norm_opes)
        h_macs = self.get_macs(curr_proc_adj, batch_idxes, (norm_opes, norm_macs)) 
    

        h_opes_pooled = h_opes.mean(dim=-2)
        h_macs_pooled = h_macs.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_opes_expand = h_opes.unsqueeze(-2).expand(-1, -1, curr_proc_adj.size(-1), -1)
        h_macs_expand = h_macs.unsqueeze(-3).expand(-1, h_opes.size(-2), -1, -1)
        h_opes_pooled_expand = h_opes_pooled[:, None, None, :].expand_as(h_opes_expand)
        h_macs_pooled_expand = h_macs_pooled[:, None, None, :].expand_as(h_macs_expand)

        h_actions = torch.cat((h_opes_expand, h_macs_expand, h_opes_pooled_expand, h_macs_pooled_expand), dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_macs_pooled), dim=-1)
        prob = self.actor(h_actions).flatten(1)
        mask = mask_proc.transpose(1, 2).flatten(1)

        prob[~mask] = float('-inf')
        action_probs = F.softmax(prob, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_indexes)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys