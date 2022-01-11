from math import e
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import warnings 
import math
import random
from torch.distributions import Categorical
from collections import OrderedDict
warnings.filterwarnings("ignore")

T.set_printoptions(sci_mode=False, threshold=1000)

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
    
    g[g != g] = 0
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
    x[x != x] = 0
    return x

class LinearDeepNetwork(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, n_actions, input_dims, hidden_size = 16):
        super(LinearDeepNetwork, self).__init__()
        self.net = nn.Sequential(OrderedDict({
            'fc_1': nn.Linear(input_dims, hidden_size),
            'act_1': nn.ReLU(),
            'fc_2': nn.Linear(hidden_size, n_actions),
            'act_2': nn.Softmax(),
        }))
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                T.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
    
        self.net.apply(init_weights)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

class LinearRDeepNetwork(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, n_actions, input_dims, hidden_size = 16):
        super(LinearRDeepNetwork, self).__init__()
        self.net = nn.Sequential(OrderedDict({
            'fc_1': nn.Linear(input_dims, hidden_size),
            #'norm_1': nn.LayerNorm(hidden_size),
            'act_1': nn.ReLU(),
            'fc_2': nn.Linear(hidden_size, n_actions),
            #'norm_2': nn.LayerNorm(n_actions),
            'act_2': nn.Sigmoid()
        }))
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                T.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        def init_zero(m):
            if isinstance(m, nn.Linear):
                T.nn.init.constant_(m.weight, 0)
                m.bias.data.zero_()

        def init_kaiming(m):
            if isinstance(m, nn.Linear):
                T.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        self.net.apply(init_weights)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

class LinearDeepNetwork_no_activation(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, n_actions, input_dims, hidden_size = 16):
        super(LinearDeepNetwork_no_activation, self).__init__()
        self.net = nn.Sequential(OrderedDict({
            'fc_1': nn.Linear(input_dims, hidden_size),
            'act_1': nn.ReLU(),
            'fc_2': nn.Linear(hidden_size, n_actions),
        }))
        
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
    def __init__(self, n_action, observation_dim, top_n, lr, lrdc, weight_decay, max_d_kl, entropy_weight, pmax):
        self.lr = lr
        self.lrdc = lrdc
        self.weight_decay = weight_decay
        self.n_action = n_action
        self.top_n = top_n
        self.loss = nn.CrossEntropyLoss()    
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.entropy_weight = entropy_weight*T.ones(1).to(self.device)
        self.policy = LinearDeepNetwork(n_actions = n_action, input_dims = (2+2*self.top_n) * observation_dim + (2) * self.top_n)
        self.params = self.policy.parameters()
        self.max_d_kl = max_d_kl
        #self.params.append(self.entropy_weight)
        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lrdc)
        self.pmax = pmax
    
    def save(self, path):
        T.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(T.load(path))
    
    def gail_update(self, all_expert_traj, all_self_traj):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        '''
        # update policy
        batch_loss = 0
        eps = 5e-4

        self_a_list = T.LongTensor([a for self_traj in all_self_traj for _,a,_ in self_traj]).to(self.device)
        self_s_list = T.stack(([s for self_traj in all_self_traj for s,_,_ in self_traj])).to(self.device)
        self_p_list = T.tensor([np.log(1-p+eps) for self_traj in all_self_traj for _,_,p in self_traj]).to(self.device)
        predicted_a_list = self.policy.forward(self_s_list).to(self.device)
        predicted_probs = predicted_a_list[range(predicted_a_list.shape[0]),self_a_list]
        L = -(T.mul(T.log(predicted_probs + eps), T.log(self_p_list+eps))).mean() + self.entropy_weight * entropy_e(predicted_a_list).to(self.device)
        #L = (T.mul(T.log(predicted_probs), self_p_list)).mean() + self.entropy_weight * entropy_e(predicted_a_list).to(self.device)
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

                L_new = -(T.mul(T.log(predicted_probs_new), T.log(self_p_list))).mean() + self.entropy_weight * entropy_e(predicted_a_list_new).to(self.device)
                #L_new = (T.mul(T.log(predicted_probs_new), self_p_list)).mean() + self.entropy_weight * entropy_e(predicted_a_list_new).to(self.device)
                KL_new = kl_div(predicted_a_list, predicted_a_list_new)

            L_improvement = L_new - L

            if L_improvement > 0 and KL_new <= self.max_d_kl:
                return True

            apply_update(-step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1
        print('updated', i)

        batch_loss += L.detach().item()

        predicted_a_list = self.policy.forward(self_s_list).to(self.device)
        predicted_probs = predicted_a_list[range(predicted_a_list.shape[0]),self_a_list]
        L = -(T.mul(T.log(predicted_probs), T.log(self_p_list+eps))).mean() + self.entropy_weight * entropy_e(predicted_a_list).to(self.device)
        #L = (T.mul(T.log(predicted_probs), self_p_list)).mean() + self.entropy_weight * entropy_e(predicted_a_list).to(self.device)
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
        for i in range(self.top_n):
            encoded_state = T.cat((encoded_state, questions_embeddings[i]), dim=0)
        for i in range(self.top_n):
            encoded_state = T.cat((encoded_state, answers_embeddings[i]), dim=0)
        encoded_state = T.cat((encoded_state, questions_scores[:self.top_n]), dim=0)
        encoded_state = T.cat((encoded_state, answers_scores[:self.top_n]), dim=0)
        state = T.tensor(encoded_state, dtype=T.float).to(self.device)
        pp = self.policy.forward(state)
        ##print("pp", pp)
        action = T.argmax(pp).item()
        #dist = Categorical(action)
        #sample_action = dist.sample().item()
        return state, action

class GAILAgent():
    '''
    The multi-objective Inverse reinforcement learning Agent for conversational search.
    This agent has multiple policies each represented by one <agent> object.
    '''
    def __init__(self, n_action, observation_dim, top_n, lr, lrdc, weight_decay, max_d_kl, entropy_weight, pmax, disc_weight_clip, policy_weight_clip, gan_name, disc_pretrain_epochs):
        self.lr = lr
        self.lrdc = lrdc
        self.weight_decay = weight_decay
        self.n_action = n_action
        self.top_n = top_n
        self.loss = nn.MSELoss()    
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.entropy_weight = entropy_weight*T.ones(1).to(self.device)
        self.observation_dim = observation_dim
        self.disc_weight_clip = disc_weight_clip
        self.policy_weight_clip = policy_weight_clip
        self.policy = LinearDeepNetwork(n_actions = n_action, input_dims = (2+2*self.top_n) * observation_dim + (2) * self.top_n)
        self.gan_name = gan_name
        self.disc_pretrain_epochs = disc_pretrain_epochs
        #self.disc = LinearDeepNetwork_no_activation(n_actions = 1, input_dims = (2+self.top_n) * observation_dim + self.top_n) if gan_name == 'WGAN' else LinearRDeepNetwork(n_actions = 1, input_dims = (2+self.top_n) * observation_dim + self.top_n)
        self.disc = LinearRDeepNetwork(n_actions = 1, input_dims = (2+self.top_n) * observation_dim + self.top_n)
        self.policyparams = self.policy.parameters()
        self.discparams = self.disc.parameters()
        self.max_d_kl = max_d_kl
        self.disc_optimizer = optim.RMSprop(self.discparams, lr=self.lr, weight_decay = self.weight_decay)
        self.policy_optimizer = optim.RMSprop(self.policyparams, lr=self.lr, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.disc_optimizer, gamma=self.lrdc)
        self.pmax = pmax
        self.expert_traj_history = []
        self.self_traj_history = []
    
    def save(self, path):
        T.save(self.policy.state_dict(), path+'_policy')
        T.save(self.disc.state_dict(), path+'_disc')
    
    def load(self, path):
        self.policy.load_state_dict(T.load(path+'_policy'))
        self.disc.load_state_dict(T.load(path+'_disc'))

    def gail_update(self, all_expert_traj, all_self_traj, disc_train_ratio, epoch):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        '''
        # update policy
        batch_loss = 0
        eps = 1e-4

        self_a_list = T.LongTensor([a for self_traj in all_self_traj for _,a,_ in self_traj]).to(self.device)
        self_s_list = T.stack(([s for self_traj in all_self_traj for s,_,_ in self_traj])).to(self.device)
        self_p_list = T.tensor([1-p for self_traj in all_self_traj for _,_,p in self_traj]).to(self.device)
        predicted_a_list = self.policy.forward(self_s_list).to(self.device)
        predicted_probs = T.max(predicted_a_list, 1).values

        L = T.zeros(1).to(self.device)

        # compute self state_action input for discriminator
        self_disc_s_a_list = []
        for i, row in enumerate(self_s_list):
            disc_context = row[:2*self.observation_dim]
            column_start = 2 * self.observation_dim if int(self_a_list[i]) > 0 else (2 + self.top_n) * self.observation_dim
            column_end = (2 + self.top_n) * self.observation_dim if int(self_a_list[i]) > 0 else (2 + 2 * self.top_n) * self.observation_dim  
            score_start = - self.top_n * (1 + int(self_a_list[i])) - 1 
            disc_candidate = T.cat((row[column_start: column_end], row[score_start: score_start + self.top_n]))
            self_disc_s_a_list.append(T.cat((disc_context, disc_candidate)))
        self_disc_s_a = T.stack(self_disc_s_a_list).to(self.device)

        # compute expert state_action input for discriminator
        expert_s_list = T.stack(([s for expert_traj in all_expert_traj for s,_ in expert_traj])).to(self.device)
        expert_a_list = T.LongTensor([a for expert_traj in all_expert_traj for _,a in expert_traj]).to(self.device)
        expert_disc_s_a_list = []
        for i, row in enumerate(expert_s_list):
            disc_context = row[:2*self.observation_dim]
            column_start = 2 * self.observation_dim if int(expert_a_list[i]) > 0  else (2 + self.top_n) * self.observation_dim
            column_end = (2 + self.top_n) * self.observation_dim if int(expert_a_list[i]) > 0  else (2 + 2 * self.top_n) * self.observation_dim
            score_start = - self.top_n * (1 + int(expert_a_list[i])) - 1 
            disc_candidate = T.cat((row[column_start: column_end], row[score_start: score_start + self.top_n]))
            expert_disc_s_a_list.append(T.cat((disc_context, disc_candidate)))
        expert_disc_s_a = T.stack(expert_disc_s_a_list).to(self.device)


        if self.gan_name == 'WGAN':
            disc_train_ratio = 5

        for di in range(disc_train_ratio):
            disc_self_p = self.disc.forward(self_disc_s_a).to(self.device)
            disc_expert_p = self.disc.forward(expert_disc_s_a).to(self.device)
            # update discriminator
            if self.gan_name == 'GAN':
                L_disc = - T.log(disc_expert_p).mean() - T.log(1-disc_self_p).mean() # log discriminator loss
            elif self.gan_name == 'LSGAN':
                L_disc = self.loss(disc_expert_p, T.tensor([1.0]*len(disc_expert_p)).to(self.device)) + self.loss(disc_self_p, T.tensor([0.0]*len(disc_self_p)).to(self.device)) # mse discriminator loss
            elif self.gan_name == 'WGAN':
                L_disc = -( disc_expert_p.mean() - disc_self_p.mean() )
            self.disc_optimizer.zero_grad()
            L_disc.backward(retain_graph = True)
            self.disc_optimizer.step()
            '''
            print(disc_expert_p, disc_self_p)
            print(L_disc)
            print(dict(self.disc.net.named_children())['fc_1'].weight.grad)
            print(dict(self.disc.net.named_children())['fc_1'].bias.grad, flush = True)
            '''
            if self.gan_name == 'WGAN':
                for p in self.disc.parameters():
                    p.data.clamp_(-self.disc_weight_clip, self.disc_weight_clip)
            
        if epoch < self.disc_pretrain_epochs:
            return L_disc.detach().item(), -1000*epoch

        # update policy
        L_pol = T.zeros(1).to(self.device)
        for k in range(len(all_self_traj)):
            expert_a = [a for _,a in all_expert_traj[k]]
            conv_a_list = T.LongTensor([a for _,a,_ in all_self_traj[k]]).to(self.device)
            conv_s_list = T.stack(([s for s,_,_ in all_self_traj[k]])).to(self.device)
            print("conv_a_list", conv_a_list)
            conv_disc_s_a_list = []
            conv_disc_s_a_list_2 = []
            for ck, row in enumerate(conv_s_list):
                disc_context = row[:2*self.observation_dim]
                column_start = 2 * self.observation_dim if int(conv_a_list[ck]) > 0 else (2 + self.top_n) * self.observation_dim
                column_end = (2 + self.top_n) * self.observation_dim if int(conv_a_list[ck]) > 0 else (2 + 2 * self.top_n) * self.observation_dim
                score_start = - self.top_n * (1 + int(conv_a_list[ck])) - 1 
                disc_candidate = T.cat((row[column_start: column_end], row[score_start: score_start + self.top_n]))
                conv_disc_s_a_list.append(T.cat((disc_context, disc_candidate)))

                column_start_2 = (2 + self.top_n) * self.observation_dim if int(conv_a_list[ck]) > 0 else 2 * self.observation_dim
                column_end_2 = (2 + 2 * self.top_n) * self.observation_dim if int(conv_a_list[ck]) > 0 else (2 + self.top_n) * self.observation_dim
                score_start_2 = - self.top_n * (2 - int(conv_a_list[ck])) - 1 
                disc_candidate_2 = T.cat((row[column_start_2: column_end_2], row[score_start_2: score_start_2 + self.top_n]))
                conv_disc_s_a_list_2.append(T.cat((disc_context, disc_candidate_2)))
            conv_disc_s_a = T.stack(conv_disc_s_a_list).to(self.device)
            conv_disc_s_a_2 = T.stack(conv_disc_s_a_list_2).to(self.device)

            for j in range(len(all_self_traj[k])):
                print("round", j)
                print(conv_a_list[j])
                distrib = self.policy.forward(all_self_traj[k][j][0]).to(self.device)

                if conv_a_list[j] > 0:
                    Qs = self.disc.forward(conv_disc_s_a[j:])
                    Q = Qs.mean()
                    Qs_2 = self.disc.forward(conv_disc_s_a_2[j])
                    Q_2 = Qs_2.mean()
                else:
                    Qs = self.disc.forward(conv_disc_s_a[j])
                    Q = Qs.mean()
                    Qs_2 = self.disc.forward(conv_disc_s_a_2[j:])
                    Q_2 = Qs_2.mean()
                    
                print(distrib, Qs, Q,  Qs_2, Q_2)
                try:
                    print(expert_a[j])
                except:
                    pass
                L_pol -= T.log(distrib[conv_a_list[j]])* Q + T.log(distrib[1-conv_a_list[j]])*Q_2 + self.entropy_weight * entropy_e([distrib]).to(self.device)
                #L_pol -= T.log(p_s_a)*Q + T.log(p_s_a_2)*Q_2 + self.entropy_weight * entropy_e([distrib]).to(self.device)
                #L_pol -= T.log(p_s_a)*Q  + self.entropy_weight * entropy_e([distrib]).to(self.device)

            
        # l1 penalty
        l1 = 0
        for p in self.policy.parameters():
            l1 += p.abs().sum()
            
        L_pol = L_pol + self.weight_decay * l1

        self.policy_optimizer.zero_grad()
        L_pol.backward()
        print(dict(self.policy.net.named_children())['fc_1'].weight.grad)
        print(dict(self.policy.net.named_children())['fc_1'].bias.grad, flush = True)
        self.policy_optimizer.step()
        
          
        for p in self.policy.parameters():
            p.data.clamp_(-self.policy_weight_clip, self.policy_weight_clip)
        
        batch_loss += L_pol.detach().item()

        return L_disc.detach().item(), batch_loss


    def inference_step(self, query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores, mode):
        '''
        The inference step of moirl agent first computes the posterior distribution of policies given existing conversation trajectory.
        Then the distribution of policies can be used to compute a weighted reward function.
        '''
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
        for i in range(self.top_n):
            encoded_state = T.cat((encoded_state, questions_embeddings[i]), dim=0)
        for i in range(self.top_n):
            encoded_state = T.cat((encoded_state, answers_embeddings[i]), dim=0)
        encoded_state = T.cat((encoded_state, questions_scores[:self.top_n]), dim=0)
        encoded_state = T.cat((encoded_state, answers_scores[:self.top_n]), dim=0)
        state = T.tensor(encoded_state, dtype=T.float).to(self.device)
        pp = self.policy.forward(state)
        if mode == 'test':
            print(pp)
        action = T.argmax(pp).item()
        return state, action
    
    def sample_step(self, query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores):
        '''
        The inference step of moirl agent first computes the posterior distribution of policies given existing conversation trajectory.
        Then the distribution of policies can be used to compute a weighted reward function.
        '''
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
        for i in range(self.top_n):
            encoded_state = T.cat((encoded_state, questions_embeddings[i]), dim=0)
        for i in range(self.top_n):
            encoded_state = T.cat((encoded_state, answers_embeddings[i]), dim=0)
        encoded_state = T.cat((encoded_state, questions_scores[:self.top_n]), dim=0)
        encoded_state = T.cat((encoded_state, answers_scores[:self.top_n]), dim=0)
        state = T.tensor(encoded_state, dtype=T.float).to(self.device)
        pp = self.policy.forward(state)
        if random.random() > pp[0]:
            action = 1
        else:
            action = 0
        return state, action


    def ecrr_update(self, all_expert_traj, all_self_traj):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        '''
        # update policy
        batch_loss = 0
        self.policy_optimizer.zero_grad()

        L = T.zeros(1).to(self.device)
        for k in range(len(all_self_traj)):
            expert_a = [a for _,a in all_expert_traj[k]]
            for j in range(len(all_self_traj[k])):
                if j < len(expert_a):
                    distrib = self.policy.forward(all_self_traj[k][j][0]).to(self.device)
                    print(all_self_traj[k][j][2], distrib, expert_a[j])
                    L -= T.log(distrib[expert_a[j]])*(1-all_self_traj[k][j][2]) + self.entropy_weight * entropy_e([distrib]).to(self.device)

        L.backward()
        self.policy_optimizer.step()
        self.scheduler.step()
        batch_loss += L.detach().item()

        return 0, batch_loss
