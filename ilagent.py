from math import e
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import warnings 
import math
from torch.distributions import Categorical
warnings.filterwarnings("ignore")

T.set_printoptions(sci_mode=False)

def entropy_e(ts):
    '''
    Assuming ts is 2d tensor
    '''
    ent = 0
    for t in ts:
        for p in t:
            ent -= p * T.log(p)
    return ent

def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = T.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = T.cat([t.view(-1) for t in g])
    return g

def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = T.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x


class LinearDeepNetwork(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, n_actions, input_dims, hidden_size = 16):
        super(LinearDeepNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(),
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                T.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.net.apply(init_weights)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.net(state)


class ILAgent():
    '''
    The multi-objective Inverse reinforcement learning Agent for conversational search.
    This agent has multiple policies each represented by one <agent> object.
    '''
    def __init__(self, n_action, observation_dim, top_n, lr, lrdc, weight_decay, max_d_kl = 0.01):
        self.lr = lr
        self.lrdc = lrdc
        self.weight_decay = weight_decay
        self.n_action = n_action
        self.top_n = top_n
        self.loss = nn.CrossEntropyLoss()    
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.entropy_weight = 1e-3*T.ones(1).to(self.device)
        self.policy = LinearDeepNetwork(n_actions = n_action, input_dims = (2) * observation_dim + (2) * self.top_n)
        self.params = self.policy.parameters()
        self.max_d_kl = max_d_kl
        #self.params.append(self.entropy_weight)
        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lrdc)
    
    def save(self, path):
        T.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(T.load(path))

    def gail_step(self, all_self_traj):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        '''
        # update policy
        batch_loss = 0
        self.optimizer.zero_grad()
        L = T.zeros(1).to(self.device)


        print("prior update")
        '''  
        for j, traj in enumerate(all_expert_traj):
            expert_a_list = T.tensor([a for s,a in traj]).to(self.device)
            expert_s_list = T.stack(([s for s,a in traj])).to(self.device)
            predicted_a_list = self.policy.forward(expert_s_list).to(self.device)
            L += self.loss(predicted_a_list, expert_a_list)
        '''
        
        for k, traj in enumerate(all_self_traj):
            self_s_a_list, p = traj
            self_a_list = T.tensor([a for s,a,_ in self_s_a_list]).to(self.device)
            self_s_list = T.stack(([s for s,a,_ in self_s_a_list])).to(self.device)
            predicted_a_list = self.policy.forward(self_s_list).to(self.device)
            L +=  -(1-p) * self.loss(predicted_a_list, self_a_list) - self.entropy_weight * entropy_e(predicted_a_list)
            print(predicted_a_list, self_a_list)

        L = L.to(self.device)
        L.backward() 
        self.optimizer.step()
        batch_loss += L.detach().item()

        print("post update")  
        for k, traj in enumerate(all_self_traj):
            self_s_a_list, p = traj
            self_a_list = T.tensor([a for s,a,_ in self_s_a_list]).to(self.device)
            self_s_list = T.stack(([s for s,a,_ in self_s_a_list])).to(self.device)
            predicted_a_list = self.policy.forward(self_s_list).to(self.device)
            print(predicted_a_list, self_a_list)
    
        return batch_loss

    
    def trpo_update(self, all_self_traj):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        '''
        # update policy
        batch_loss = 0

        self_a_list = T.LongTensor([a for self_s_a_list,_ in all_self_traj for s,a,_ in self_s_a_list]).to(self.device)
        self_s_list = T.stack(([s for self_s_a_list,_ in all_self_traj for s,a,_ in self_s_a_list])).to(self.device)
        self_p_list = T.tensor([(1-p) for self_s_a_list,p in all_self_traj for s,a,_ in self_s_a_list]).to(self.device)
        predicted_a_list = self.policy.forward(self_s_list).to(self.device)
        predicted_probs = predicted_a_list[range(predicted_a_list.shape[0]),self_a_list]
        L = -(T.mul(T.log(predicted_probs), self_p_list)).mean() + self.entropy_weight * entropy_e(predicted_a_list).to(self.device)
        print('L ', L)
        print("prior update")
        print(predicted_a_list, self_a_list, self_p_list)

        KL = kl_div(predicted_a_list, predicted_a_list)
        parameters = list(self.policy.parameters())
        g = flat_grad(L, parameters, retain_graph=True)
        d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

        def HVP(v):
            return flat_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = conjugate_gradient(HVP, g)
        max_length = T.sqrt(2 * self.max_d_kl / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir
        
        def apply_update(grad_flattened):
            n = 0
            for p in parameters:
                numel = p.numel()
                g = grad_flattened[n:n + numel].view(p.shape)
                p.data += g
                n += numel

        def criterion(step):
            apply_update(step)

            with T.no_grad():
                predicted_a_list_new = self.policy.forward(self_s_list).to(self.device)
                predicted_probs_new = predicted_a_list_new[range(predicted_a_list.shape[0]),self_a_list]

                L_new = -(T.mul(T.log(predicted_probs_new), self_p_list)).mean() + self.entropy_weight * entropy_e(predicted_a_list_new).to(self.device)
                KL_new = kl_div(predicted_a_list, predicted_a_list_new)

            L_improvement = L_new - L

            if L_improvement > 0 and KL_new <= self.max_d_kl:
                return True

            apply_update(-step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1

        batch_loss += L.detach().item()

        self_a_list = T.LongTensor([a for self_s_a_list,_ in all_self_traj for s,a,_ in self_s_a_list]).to(self.device)
        self_s_list = T.stack(([s for self_s_a_list,_ in all_self_traj for s,a,_ in self_s_a_list])).to(self.device)
        self_p_list = T.tensor([(1-p) for self_s_a_list,p in all_self_traj for s,a,_ in self_s_a_list]).to(self.device)
        predicted_a_list = self.policy.forward(self_s_list).to(self.device)
        predicted_probs = predicted_a_list[range(predicted_a_list.shape[0]),self_a_list]
        L = -(T.mul(T.log(predicted_probs), self_p_list)).mean() + self.entropy_weight * entropy_e(predicted_a_list).to(self.device)
        print('L ', L)
        print("post update")  
        print(predicted_a_list, self_a_list, self_p_list)
        return batch_loss


    def inference_step(self, query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores):
        '''
        The inference step of moirl agent first computes the posterior distribution of policies given existing conversation trajectory.
        Then the distribution of policies can be used to compute a weighted reward function.
        '''
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
        #encoded_state = T.cat((encoded_state, encoded_q), dim=0)
        #encoded_state = T.cat((encoded_state, answers_embeddings[0]), dim=0)
        encoded_state = T.cat((encoded_state, questions_scores[:self.top_n]), dim=0)
        encoded_state = T.cat((encoded_state, answers_scores[:self.top_n]), dim=0)
        state = T.tensor(encoded_state, dtype=T.float).to(self.device)
        pp = self.policy.forward(state)
        action = T.argmax(pp).item()
        #dist = Categorical(action)
        #sample_action = dist.sample().item()
        return state, action
