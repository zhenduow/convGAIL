from user import User
from dataset import ConversationDataset
from agent import Agent, BaseAgent
from moirlagent import MOIRLAgent
import logging
import numpy as np
import random
import os
import torch as T
from transformers import AutoTokenizer, AutoModel
from scipy.special import softmax
from parlai.scripts.interactive import Interactive, rerank
from copy import deepcopy
import argparse
import gc

def generate_embedding_no_grad(text, tokenizer, embedding_model, device):
    '''
    Generate embedding using torch transformer.
    '''
    with T.no_grad():
        tokenized_context_ = T.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
        context_embedding_ = T.squeeze(embedding_model(tokenized_context_)[0])[0].detach().cpu()
        del tokenized_context_
        T.cuda.empty_cache()
        gc.collect()
        return context_embedding_

def read_from_memory(query, context, memory):
    '''
    Read query, context, question, answer ranks and ranking scores from memory
    '''
    return memory[query]['embedding'], memory[query][context]['embedding'],\
        memory[query][context]['questions'], memory[query][context]['answers'],\
        memory[query][context]['questions_embeddings'],memory[query][context]['answers_embeddings'],\
        memory[query][context]['questions_scores'], memory[query][context]['answers_scores']

def save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device):
    '''
    Save query, context, question, answer ranks and ranking scores to memory for running speed.
    '''
    if query not in memory.keys():
        memory[query] = {}
        with T.no_grad():
            tokenized_query = T.tensor([tokenizer.encode(query, add_special_tokens=True)]).to(device)
            memory[query]['embedding'] = T.squeeze(embedding_model(tokenized_query)[0])[0].detach().cpu()
            T.cuda.empty_cache()
    memory[query][context] = {}
    with T.no_grad():
        memory[query][context]['embedding'] = T.squeeze(embedding_model(T.tensor([tokenizer.encode(context, add_special_tokens=True)]).to(device))[0])[0].detach().cpu()
        memory[query][context]['questions_embeddings'] = [T.squeeze(embedding_model(T.tensor([tokenizer.encode(questions[i], add_special_tokens=True)]).to(device))[0])[0].detach().cpu() for i in range(3)]
        memory[query][context]['answers_embeddings'] = [T.squeeze(embedding_model(T.tensor([tokenizer.encode(answers[0], add_special_tokens=True)]).to(device))[0])[0].detach().cpu()]
        memory[query][context]['questions'] = questions
        memory[query][context]['answers'] = answers
        memory[query][context]['questions_scores'] = T.tensor(questions_scores).detach().cpu()
        memory[query][context]['answers_scores'] = T.tensor(answers_scores).detach().cpu()
        T.cuda.empty_cache()
    gc.collect()
    return memory

def generate_batch_question_candidates(batch, conversation_id, ignore_questions, total_candidates):
    '''
    Generate positive and negative clarifying question answers.
    '''
    positives = [batch['conversations'][conversation_id][turn_id] for turn_id in range(len(batch['conversations'][conversation_id])) if turn_id % 2 == 1 and turn_id != len(batch['conversations'][conversation_id])-1]
    filtered_positives = [cand for cand in positives if cand not in ignore_questions]
    negatives = [response for response in batch['responses_pool'] if response not in positives][:total_candidates - len(filtered_positives)]
    return filtered_positives + negatives

def generate_batch_answer_candidates(batch, conversation_id, total_candidates):
    '''
    Generate positive and negative answer candidates.
    '''
    positives = [batch['conversations'][conversation_id][-1]]
    negatives = [answer for answer in batch['answers_pool'] if answer not in positives][:total_candidates - len(positives)] 
    return positives + negatives

def compute_efficiency(reciprocal_rank, round):
    '''
    The intuition behind this metric is to split the range of 1/r and 1/(r+1), where 1/r = reciprocal_rank.
    The efficiency score for 1 round maps to 1/r, and the efficiency score for infinite round maps to 1/(r+1).
    In another word, quality is prioritized.
    '''
    return (float(reciprocal_rank) + float(reciprocal_rank)*float(reciprocal_rank) / float(round))/(float(reciprocal_rank) + 1)

def create_rerankers(dataset_name, reranker_name):
    '''
    Create rerankers.
    '''
    if dataset_name == 'MSDialog':
        reranker_prefix = ''
    elif dataset_name == 'UDC':
        reranker_prefix = 'udc'
    elif dataset_name == 'Opendialkg':
        reranker_prefix = 'open'
    
    question_reranker = Interactive.main(model = 'transformer/' + reranker_name + 'encoder', 
                    model_file = 'zoo:pretrained_transformers/model_'+ reranker_name + '/' + reranker_prefix + 'question',  
                    encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                    return_cand_scores = True)

    answer_reranker = Interactive.main(model = 'transformer/' + reranker_name + 'encoder', 
                    model_file = 'zoo:pretrained_transformers/model_' + reranker_name + '/' + reranker_prefix + 'answer',  
                    encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                    return_cand_scores = True)

    return question_reranker, answer_reranker

def initialize_dirs(dataset_name, reranker_name, cv):
    '''
    Create folders for saving cache, checkpoints, and logs.
    '''
    if not os.path.exists(dataset_name + '_experiments/checkpoints/'):
        os.makedirs(dataset_name + '_experiments/checkpoints/')
    if not os.path.exists(dataset_name + '_experiments/embedding_cache/'):
        os.makedirs(dataset_name + '_experiments/embedding_cache/')
    if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name ):
        os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name )
    if cv != '':
        if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv):
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv)
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv + '/train')
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv + '/val')
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv + '/test')
    else:
        if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/train' ):
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/train')
        if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/val' ):
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/val' )
        if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/test' ):
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/test' )

def find_best_trajectory(answer_traj, question_traj, p):
    '''
    Find the best conversation trajectory given all the answer rank and question rank.
    answer_traj is a list of (state, action, reward) tuples.
    question_traj is a list of (state, action, correct question rank) tuples.
    '''
    best_ecrr, best_step = 0, 0
    cumulative_ecrr = 1
    assert len(answer_traj) == len(question_traj)
    for step in range(len(answer_traj)):
        if cumulative_ecrr * answer_traj[step][1] > best_ecrr:
            best_ecrr = cumulative_ecrr * answer_traj[step][1]
            best_step = step
        cumulative_ecrr *= p ** question_traj[step][1]
    return best_step

def compute_trajectory_ecrr(traj, continue_p):
    p = 1
    for s, a, r in traj:
        if a == 0:
            p *= r
            break
        else:
            p *= continue_p ** r
    return p


def main(args):
    logging.getLogger().setLevel(logging.INFO)
    random.seed(2020)
    device = "cuda:0" if T.cuda.is_available() else "cpu"

    # load data
    train_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/train' + args.cv + '/', args.batch_size, args.max_data_size)
    val_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/val' + args.cv + '/', args.batch_size, args.max_data_size)
    test_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/test' + args.cv + '/', args.batch_size, args.max_data_size)
    data_size = sum([len(b['conversations'].keys()) for b in train_dataset.batches]) 
    test_size = sum([len(b['conversations'].keys()) for b in test_dataset.batches]) 

    # load rerankers
    question_reranker, answer_reranker = create_rerankers(args.dataset_name, args.reranker_name)

    # initialize agents
    agent = Agent(lr = 1e-4, input_dims = (3 + args.topn) * args.observation_dim + 1 + args.topn, top_k = args.topn, n_actions=args.n_action, gamma = 1 - args.cq_reward, weight_decay = args.weight_decay) # query, context, answer, and topn questions embedding + 1 answer score and topn question score
    base_agent = BaseAgent(lr = 1e-4, input_dims = 2 * args.observation_dim, n_actions = args.n_action, weight_decay = args.weight_decay)
    moirlagent = MOIRLAgent(n_policy = args.n_policy, n_action = args.n_action, observation_dim = args.observation_dim, top_n = args.topn, lr= args.lr, weight_decay= args.weight_decay)
    
    # initialize embedding model
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
    embedding_model = AutoModel.from_pretrained('xlnet-base-cased').to(device)

    # initialize log directories
    initialize_dirs(args.dataset_name, args.reranker_name, args.cv)
    
    if args.load_checkpoint == True:
        agent.load(args.dataset_name + '_experiments/checkpoints/' + args.checkpoint)
        base_agent.load(args.dataset_name + '_experiments/checkpoints/' + args.checkpoint_base)
        moirlagent.load(args.dataset_name + '_experiments/checkpoints/' + args.checkpoint_moirl)
    
    for i in range(args.train_iter):
        train_scores, train_q0_scores, train_q1_scores, train_q2_scores, train_oracle_scores, train_base_scores, train_moirl_scores  = [],[],[],[],[],[],[]
        train_worse, train_q0_worse, train_q1_worse, train_q2_worse, train_base_worse, train_moirl_worse = [],[],[],[],[],[]
        moirl_loss = 0
        for batch_serial, batch in enumerate(train_dataset.batches):
            all_pos_traj = [] # all conversation trajectories used for MOIRL
            all_neg_traj = []
            all_self_traj = []
            # load state information from cached memory
            if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/train/memory.batchsave' + str(batch_serial)):
                with T.no_grad():
                    memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/train/memory.batchsave' + str(batch_serial))
            else:
                memory = {}
            
            train_ids = list(batch['conversations'].keys())
            user = User(batch['conversations'], cq_reward = args.cq_reward, cq_penalty = args.cq_reward - 1)
            for conv_serial, train_id in enumerate(train_ids):
                query = user.initialize_state(train_id)
                if query == '': # UDC dataset has some weird stuff
                    continue
                context = ''
                ignore_questions = []
                n_round = 0
                patience_used = 0
                q_done = False
                a_traj, q_traj, moirl_traj = [], [], []
                stop, base_stop, moirl_stop = False, False, False
                correct_question_rank = 0
                print('-------- train batch %.0f conversation %.0f/%.0f --------' % (batch_serial, args.batch_size*(batch_serial) + conv_serial + 1, data_size))
                
                #while not q_done:
                while n_round < len(batch['conversations'][train_id]) / 2 and correct_question_rank <= args.reranker_return_length:
                    print('-------- round %.0f --------' % (n_round))
                    if query in memory.keys():
                        if context not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, train_id, args.batch_size)
                            # get reranker results   
                            questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                            memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                            
                    else:
                        # sampling
                        question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, args.batch_size)
                        answer_candidates = generate_batch_answer_candidates(batch, train_id, args.batch_size)
                        # get reranker results   
                        questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                        answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                        memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                    
                    query_embedding, context_embedding, questions, answers, questions_embeddings, answers_embeddings, questions_scores, answers_scores = read_from_memory(query, context, memory)
                    state, action = agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores)
                    base_action = base_agent.choose_action(query_embedding, context_embedding)
                    _, moirl_action = moirlagent.inference_step(moirl_traj, state)
                   
                    context_, answer_reward, question_reward, correct_question_rank, q_done, good_question, patience_this_turn = user.update_state(train_id, context, questions, answers, use_top_k = max(args.topn - patience_used, 1))
                    patience_used = max(patience_used + patience_this_turn, args.topn)
                    print('act', action, 'base act', base_action, 'moirl act', moirl_action, 'a reward', answer_reward, 'q reward', question_reward, 'cq rank', correct_question_rank)
                    
                    a_traj.append((state, answer_reward))
                    q_traj.append((state, correct_question_rank))
                    moirl_traj.append((state, moirl_action,[answer_reward, correct_question_rank][moirl_action]))

                    if n_round >= args.user_patience:
                        q_done = True

                    if not q_done:
                        ignore_questions.append(good_question)
                        if context_ not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, train_id, args.batch_size)
                            # get reranker results
                            questions_, questions_scores_ = rerank(question_reranker, query, context_, question_candidates)
                            answers_, answers_scores_ = rerank(answer_reranker, query, context_, answer_candidates)
                            
                            memory = save_to_memory(query, context_, memory, questions_, answers_, questions_scores_, answers_scores_, tokenizer, embedding_model, device)
                        query_embedding, context_embedding_, questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = read_from_memory(query, context_, memory)

                    else:
                        context_embedding_ = generate_embedding_no_grad(context_, tokenizer, embedding_model, device)
                        questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = None, None, None, None, None, None

                    agent.joint_learn((query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores),\
                        answer_reward, question_reward,\
                        (query_embedding, context_embedding_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_))
                    base_agent.learn(query_embedding, context_embedding, 0 if (n_round + 1) == len(user.dataset[train_id])/2 else 1)

                    # non-deterministic methods evaluation
                    if not stop:
                        train_worse.append(1 if (action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (action == 1  and question_reward == args.cq_reward - 1) else 0)
                        if (action == 0 or (action == 1 and question_reward == args.cq_reward - 1)):
                            stop = True 
                            train_scores.append(answer_reward if action == 0 else 0)
                    

                    if not base_stop:
                        train_base_worse.append(1 if (base_action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (base_action == 1  and question_reward == args.cq_reward - 1) else 0)
                        if (base_action == 0 or (base_action == 1 and question_reward == args.cq_reward - 1)):
                            base_stop = True
                            train_base_scores.append(answer_reward if base_action == 0 else 0)

                    
                    if not moirl_stop:
                        train_moirl_worse.append(1 if (moirl_action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (moirl_action == 1  and question_reward == args.cq_reward - 1) else 0)
                        if (moirl_action == 0 or (moirl_action == 1 and question_reward == args.cq_reward - 1)):
                            moirl_stop = True 
                            train_moirl_scores.append(answer_reward if moirl_action == 0 else 0)
                    
                    # deterministic methods evaluation and store optimal trajectory 
                    if n_round == 0:
                        train_q0_scores.append(answer_reward)
                        train_q0_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)
                        if q_done:
                            train_q1_scores.append(0)
                            train_q2_scores.append(0)
                            train_q1_worse.append(1)
                            train_q2_worse.append(1)
                    elif n_round == 1:
                        train_q1_scores.append(answer_reward)
                        train_q1_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)
                        if q_done:
                            train_q2_scores.append(0)
                            train_q2_worse.append(1)
                    elif n_round == 2:
                        train_q2_scores.append(answer_reward)
                        train_q2_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)

                    context = context_
                    n_round += 1
                
                best_answer_step = find_best_trajectory(a_traj, q_traj, args.cascade_p)
                best_answer_reward = a_traj[best_answer_step][1]
                pos_traj = [(q_traj[0][0],1) for s in range(0, best_answer_step)] + [(a_traj[best_answer_step][0], 0)]

                train_oracle_scores.append(best_answer_reward)
                
                self_traj = []
                # create self trajectories
                for step in range(len(moirl_traj)):
                    self_traj.append((moirl_traj[step]))
                    if int(moirl_traj[step][1]) == 0:
                        break
                
                # add positive and self trajectory to batch
                all_pos_traj.append(pos_traj)
                all_self_traj.append((self_traj, compute_trajectory_ecrr(self_traj, args.cascade_p)))

            # Use EM algorithm to update parameters for MOIRL agent, did this in batch, now for each example
            moirlagent.E_step(all_pos_traj)
            print(all_self_traj)
            print(type(all_self_traj))
            batch_loss = moirlagent.gail_step(all_self_traj)
            moirl_loss += batch_loss
            
            # save memory per batch
            T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/train/memory.batchsave' + str(batch_serial))
            del memory

            T.cuda.empty_cache()


        print("Train epoch", i)
        print("risk\tacc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_scores]), np.mean(train_scores), np.mean(train_worse)))
        print("q0\tacc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q0_scores]), np.mean(train_q0_scores), np.mean(train_q0_worse)))
        print("q1\tacc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q1_scores]), np.mean(train_q1_scores), np.mean(train_q1_worse)))
        print("q2\tacc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q2_scores]), np.mean(train_q2_scores), np.mean(train_q2_worse)))
        print("base\tacc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_base_scores]), np.mean(train_base_scores), np.mean(train_base_worse)))
        print("moirl\tacc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_moirl_scores]), np.mean(train_moirl_scores), np.mean(train_moirl_worse)))
        print("oracle\tacc %.6f, avgmrr %.6f, worse decisions 0" % 
            (np.mean([1 if score == 1 else 0 for score in train_oracle_scores]), np.mean(train_oracle_scores)))

        print("avg loss", np.mean(agent.loss_history))
        print("moirl loss", moirl_loss)

        agent.save(args.dataset_name + '_experiments/checkpoints/' + str(np.mean(agent.loss_history)))
        base_agent.save(args.dataset_name + '_experiments/checkpoints/' + 'base')
        moirlagent.save(args.dataset_name + '_experiments/checkpoints/' + 'moirl_' + str(moirl_loss))

        ## val
        val_scores, val_q0_scores, val_q1_scores, val_q2_scores, val_oracle_scores, val_base_scores, val_moirl_scores = [],[],[],[],[],[],[]
        val_worse, val_q0_worse, val_q1_worse, val_q2_worse, val_base_worse,val_moirl_worse = [],[],[],[],[],[]
        val_efficiency, val_oracle_efficiency, val_base_efficiency, val_moirl_efficiency = [], [], [], []
        val_ecrr, val_q0_ecrr, val_q1_ecrr, val_q2_ecrr, val_oracle_ecrr, val_base_ecrr, val_moirl_ecrr = [],[],[],[],[],[],[]
        agent.epsilon = 0
        
        for batch_serial, batch in enumerate(val_dataset.batches):
            if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/val/memory.batchsave' + str(batch_serial)):
                with T.no_grad():
                    memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/val/memory.batchsave' + str(batch_serial))
            else:
                memory = {}

            val_ids = list(batch['conversations'].keys())
            user = User(batch['conversations'], cq_reward = args.cq_reward, cq_penalty = args.cq_reward - 1)
            for conv_serial, val_id in enumerate(val_ids):   
                query = user.initialize_state(val_id)
                if query == '': # UDC dataset has some weird stuff
                    continue
                context = ''
                ignore_questions = []
                n_round = 0
                patience_used = 0
                q_done = False
                stop, base_stop, moirl_stop = False,False,False
                a_traj, q_traj, moirl_traj = [], [], []
                ecrr, base_ecrr, moirl_ecrr = 1, 1, 1
                cur_traj = [] # current conversation trajectory used for computing policy distribution
                correct_question_rank = 0
                print('-------- val batch %.0f conversation %.0f/%.0f --------' % (batch_serial, args.batch_size*(batch_serial) + conv_serial + 1, test_size))
                # while not q_done:
                while n_round < len(batch['conversations'][val_id]) / 2 and correct_question_rank <= args.reranker_return_length:
                    print('-------- round %.0f --------' % (n_round))
                    if query in memory.keys():
                        if context not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, val_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, val_id, args.batch_size)
                            # get reranker results   
                            questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)

                            memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                            
                    else:
                        # sampling
                        question_candidates = generate_batch_question_candidates(batch, val_id, ignore_questions, args.batch_size)
                        answer_candidates = generate_batch_answer_candidates(batch, val_id, args.batch_size)

                        # get reranker results
                        questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                        answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                    
                        memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                    
                    query_embedding, context_embedding, questions, answers, questions_embeddings, answers_embeddings, questions_scores, answers_scores = read_from_memory(query, context, memory)
                    state, action = agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores)
                    base_action = base_agent.choose_action(query_embedding, context_embedding)
                    # convMOIRL
                    state, moirl_action = moirlagent.inference_step(cur_traj, state)

                    context_, answer_reward, question_reward, correct_question_rank, q_done, good_question, patience_this_turn = user.update_state(val_id, context, questions, answers, use_top_k = max(args.topn - patience_used, 1))
                    patience_used = max(patience_used + patience_this_turn, args.topn)
                    cur_traj.append((state, moirl_action)) # update current conversation trajectory
                    print('act', action, 'moirl act', moirl_action, 'base act', base_action, 'a reward', answer_reward, 'q reward', question_reward, 'cq rank', correct_question_rank)

                    a_traj.append((state, answer_reward))
                    q_traj.append((state, correct_question_rank))
                    moirl_traj.append((state, moirl_action))

                    if n_round >= args.user_patience:
                        q_done = True

                    if not q_done:
                        ignore_questions.append(good_question)
                        if context_ not in memory[query].keys():
                            # sampling    
                            question_candidates = generate_batch_question_candidates(batch, val_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, val_id, args.batch_size)
                            # get reranker results
                            questions_, questions_scores_ = rerank(question_reranker, query, context_, question_candidates)
                            answers_, answers_scores_ = rerank(answer_reranker, query, context_, answer_candidates)
                            
                            memory = save_to_memory(query, context_, memory, questions_, answers_, questions_scores_, answers_scores_, tokenizer, embedding_model, device)
                        query_embedding, context_embedding_, questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = read_from_memory(query, context_, memory)

                    # non-deterministic models evaluation
                    if not stop:
                        val_worse.append(1 if (action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (action == 1  and question_reward == args.cq_reward - 1) else 0)
                        if (action == 0 or (action == 1 and question_reward == args.cq_reward - 1)):
                            stop = True
                            val_scores.append(answer_reward if action == 0 else 0)
                            val_efficiency.append(compute_efficiency(answer_reward, n_round + 1) if action ==0 else 0)

                    if not base_stop:
                        val_base_worse.append(1 if (base_action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (base_action == 1  and question_reward == args.cq_reward - 1) else 0)
                        if (base_action == 0 or (base_action == 1 and question_reward == args.cq_reward - 1)):
                            base_stop = True
                            val_base_scores.append(answer_reward if base_action == 0 else 0)
                            val_base_efficiency.append(compute_efficiency(answer_reward, n_round + 1) if base_action ==0 else 0)

                    if not moirl_stop:
                        val_moirl_worse.append(1 if (moirl_action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (moirl_action == 1  and question_reward == args.cq_reward - 1) else 0)
                        if (moirl_action == 0 or (moirl_action == 1 and question_reward == args.cq_reward - 1)):
                            moirl_stop = True
                            val_moirl_scores.append(answer_reward if moirl_action == 0 else 0)
                            val_moirl_efficiency.append(compute_efficiency(answer_reward, n_round + 1) if moirl_action ==0 else 0)

                    # baseline models evaluations
                    if n_round == 0:
                        val_q0_scores.append(answer_reward)
                        val_q0_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)
                        val_q0_ecrr.append(answer_reward)
                        val_q1_ecrr.append(args.cascade_p**correct_question_rank)
                        val_q2_ecrr.append(args.cascade_p**correct_question_rank)
                        if q_done:
                            val_q1_scores.append(0)
                            val_q2_scores.append(0)
                            val_q1_worse.append(1)
                            val_q2_worse.append(1)
                    elif n_round == 1:
                        val_q1_scores.append(answer_reward)
                        val_q1_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)
                        val_q1_ecrr[-1] *= answer_reward
                        val_q2_ecrr[-1] *= args.cascade_p**correct_question_rank
                        if q_done:
                            val_q2_scores.append(0)
                            val_q2_worse.append(1)
                    elif n_round == 2:
                        val_q2_scores.append(answer_reward)
                        val_q2_ecrr[-1] *= answer_reward
                        val_q2_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)

                    # ecrr evaluation
                    if 'ecrr' in locals():
                        if action == 0:
                            ecrr *= answer_reward
                            val_ecrr.append(ecrr)
                            del ecrr
                        else:
                            ecrr *= args.cascade_p**correct_question_rank
                    
                    if 'base_ecrr' in locals():
                        if base_action == 0:
                            base_ecrr *= answer_reward
                            val_base_ecrr.append(base_ecrr)
                            del base_ecrr
                        else:
                            base_ecrr *= args.cascade_p**correct_question_rank
                    
                    if 'moirl_ecrr' in locals():
                        if moirl_action == 0:
                            moirl_ecrr *= answer_reward
                            val_moirl_ecrr.append(moirl_ecrr)
                            del moirl_ecrr
                        else:
                            moirl_ecrr *= args.cascade_p**correct_question_rank

                    n_round += 1
                    context = context_

                # find the optimal trajectory
                best_answer_step = find_best_trajectory(a_traj, q_traj, args.cascade_p)
                best_answer_reward = a_traj[best_answer_step][1]
                val_oracle_scores.append(best_answer_reward)
                val_oracle_ecrr.append(best_answer_reward)

            # save batch cache
            T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/val/memory.batchsave' + str(batch_serial))
            del memory
            T.cuda.empty_cache()


        print("Val epoch", i)
        print("risk\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in val_scores]), np.mean(val_scores), np.mean(val_ecrr), np.mean(val_efficiency), np.mean(val_worse)))
        print("q0\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in val_q0_scores]), np.mean(val_q0_scores), np.mean(val_q0_ecrr), np.mean(val_q0_scores), np.mean(val_q0_worse)))
        print("q1\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in val_q1_scores]), np.mean(val_q1_scores), np.mean(val_q1_ecrr), np.mean([compute_efficiency(s, 2) for s in val_q1_scores]), np.mean(val_q1_worse)))
        print("q2\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in val_q2_scores]), np.mean(val_q2_scores), np.mean(val_q2_ecrr), np.mean([compute_efficiency(s, 3) for s in val_q2_scores]), np.mean(val_q2_worse)))
        print("base\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in val_base_scores]), np.mean(val_base_scores), np.mean(val_base_ecrr), np.mean(val_base_efficiency), np.mean(val_base_worse)))
        print("moirl\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in val_moirl_scores]), np.mean(val_moirl_scores), np.mean(val_moirl_ecrr), np.mean(val_moirl_efficiency), np.mean(val_moirl_worse)))
        print("oracle\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions 0" % 
            (np.mean([1 if score == 1 else 0 for score in val_oracle_scores]), np.mean(val_oracle_scores), np.mean(val_oracle_ecrr), np.mean(val_oracle_efficiency)))
        

        ## test
        test_scores, test_q0_scores, test_q1_scores, test_q2_scores, test_oracle_scores, test_base_scores, test_moirl_scores = [],[],[],[],[],[],[]
        test_worse, test_q0_worse, test_q1_worse,test_q2_worse, test_base_worse, test_moirl_worse = [],[],[],[],[],[]
        test_efficiency, test_oracle_efficiency, test_base_efficiency, test_moirl_efficiency = [], [], [], []
        test_ecrr, test_q0_ecrr, test_q1_ecrr, test_q2_ecrr, test_oracle_ecrr, test_base_ecrr, test_moirl_ecrr = [],[],[],[],[],[],[]
        agent.epsilon= 0
        
        for batch_serial, batch in enumerate(test_dataset.batches):
            if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/test/memory.batchsave' + str(batch_serial)):
                with T.no_grad():
                    memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/test/memory.batchsave' + str(batch_serial))
            else:
                memory = {}
            
            test_ids = list(batch['conversations'].keys())
            user = User(batch['conversations'], cq_reward = args.cq_reward, cq_penalty = args.cq_reward - 1)
            for conv_serial, test_id in enumerate(test_ids):
                query = user.initialize_state(test_id)
                if query == '': # UDC dataset has some weird stuff
                    continue
                context = ''
                ignore_questions = []
                n_round = 0
                patience_used = 0
                q_done = False
                stop, base_stop, moirl_stop = False,False,False
                ecrr, base_ecrr, moirl_ecrr = 1, 1, 1
                a_traj, q_traj, moirl_traj = [], [], []
                cur_traj = []
                correct_question_rank = 0
                print('-------- test batch %.0f conversation %.0f/%.0f --------' % (batch_serial, args.batch_size*(batch_serial) + conv_serial + 1, test_size))
                # while not q_done:
                while n_round < len(batch['conversations'][test_id]) / 2 and correct_question_rank <= args.reranker_return_length:
                    print('-------- round %.0f --------' % (n_round))
                    if query in memory.keys():
                        if context not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, test_id, args.batch_size)
                            # get reranker results   
                            questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)

                            memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                            
                    else:
                        # sampling
                        question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, args.batch_size)
                        answer_candidates = generate_batch_answer_candidates(batch, test_id, args.batch_size)
                        # get reranker results
                        questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                        answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                    
                        memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                    query_embedding, context_embedding, questions, answers, questions_embeddings, answers_embeddings, questions_scores, answers_scores = read_from_memory(query, context, memory)
                    state, action = agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores)
                    base_action = base_agent.choose_action(query_embedding, context_embedding)                    
                    # convMOIRL
                    state, moirl_action = moirlagent.inference_step(cur_traj, state)
                    context_, answer_reward, question_reward, correct_question_rank, q_done, good_question, patience_this_turn = user.update_state(test_id, context, questions, answers, use_top_k = max(args.topn - patience_used, 1))
                    patience_used = max(patience_used + patience_this_turn, args.topn)
                    
                    cur_traj.append((state, moirl_action)) # update current conversation trajectory
                    print('act', action, 'moirl act', moirl_action, 'base act', base_action, 'a reward', answer_reward, 'q reward', question_reward, 'cq rank', correct_question_rank)

                    a_traj.append((state, answer_reward))
                    q_traj.append((state, correct_question_rank))
                    moirl_traj.append((state, moirl_action))

                    if n_round >= args.user_patience:
                        q_done = True

                    if not q_done:
                        ignore_questions.append(good_question)
                        if context_ not in memory[query].keys():
                            # sampling    
                            question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, test_id, args.batch_size)
                            # get reranker results
                            questions_, questions_scores_ = rerank(question_reranker, query, context_, question_candidates)
                            answers_, answers_scores_ = rerank(answer_reranker, query, context_, answer_candidates)
                            
                            memory = save_to_memory(query, context_, memory, questions_, answers_, questions_scores_, answers_scores_, tokenizer, embedding_model, device)
                        query_embedding, context_embedding_, questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = read_from_memory(query, context_, memory)
                    # non-deterministic models evaluation
                    if not stop:
                        test_worse.append(1 if (action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (action == 1  and question_reward == args.cq_reward - 1) else 0)                    
                        if (action == 0 or (action == 1 and question_reward == args.cq_reward - 1)):
                            stop = True
                            test_scores.append(answer_reward if action == 0 else 0)
                            test_efficiency.append(compute_efficiency(answer_reward, n_round + 1) if action ==0 else 0)

                    if not base_stop:
                        test_base_worse.append(1 if (base_action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (base_action == 1  and question_reward == args.cq_reward - 1) else 0)
                        if (base_action == 0 or (base_action == 1 and question_reward == args.cq_reward - 1)):
                            base_stop = True
                            test_base_scores.append(answer_reward if base_action == 0 else 0)
                            test_base_efficiency.append(compute_efficiency(answer_reward, n_round + 1) if action ==0 else 0)
                    
                    if not moirl_stop:
                        test_moirl_worse.append(1 if (moirl_action == 0 and answer_reward < float(1/args.topn) and question_reward == args.cq_reward) \
                            or (moirl_action == 1  and question_reward == args.cq_reward - 1) else 0)
                        if (moirl_action == 0 or (moirl_action == 1 and question_reward == args.cq_reward - 1)):
                            moirl_stop = True
                            test_moirl_scores.append(answer_reward if moirl_action == 0 else 0)
                            test_moirl_efficiency.append(compute_efficiency(answer_reward, n_round + 1) if action ==0 else 0)

                    # baseline models evaluations
                    if n_round == 0:
                        test_q0_scores.append(answer_reward)
                        test_q0_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)
                        test_q0_ecrr.append(answer_reward)
                        test_q1_ecrr.append(args.cascade_p**correct_question_rank)
                        test_q2_ecrr.append(args.cascade_p**correct_question_rank)
                        if q_done:
                            test_q1_scores.append(0)
                            test_q2_scores.append(0)
                            test_q1_worse.append(1)
                            test_q2_worse.append(1)
                    elif n_round == 1:
                        test_q1_scores.append(answer_reward)
                        test_q1_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)
                        test_q1_ecrr[-1] *= answer_reward
                        test_q2_ecrr[-1] *= args.cascade_p**correct_question_rank
                        if q_done:
                            test_q2_scores.append(0)
                            test_q2_worse.append(1)
                    elif n_round == 2:
                        test_q2_scores.append(answer_reward)
                        test_q2_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == args.cq_reward else 0)
                        test_q2_ecrr[-1] *= answer_reward

                    # ecrr evaluation
                    if 'ecrr' in locals():
                        if action == 0:
                            ecrr *= answer_reward
                            test_ecrr.append(ecrr)
                            del ecrr
                        else:
                            ecrr *= args.cascade_p**correct_question_rank
                    
                    if 'base_ecrr' in locals():
                        if base_action == 0:
                            base_ecrr *= answer_reward
                            test_base_ecrr.append(base_ecrr)
                            del base_ecrr
                        else:
                            base_ecrr *= args.cascade_p**correct_question_rank
                    
                    if 'moirl_ecrr' in locals():
                        if moirl_action == 0:
                            moirl_ecrr *= answer_reward
                            test_moirl_ecrr.append(moirl_ecrr)
                            del moirl_ecrr
                        else:
                            moirl_ecrr *= args.cascade_p**correct_question_rank


                    n_round += 1
                    context = context_
                # find the optimal trajectory
                best_answer_step = find_best_trajectory(a_traj, q_traj, args.cascade_p)
                best_answer_reward = a_traj[best_answer_step][1]
                test_oracle_scores.append(best_answer_reward)
                test_oracle_ecrr.append(best_answer_reward)

            # save batch cache
            T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/test/memory.batchsave' + str(batch_serial))
            del memory
            T.cuda.empty_cache()

        
        print("Test epoch", i)
        print("risk\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_scores]), np.mean(test_scores), np.mean(test_ecrr), np.mean(test_efficiency), np.mean(test_worse)))
        print("q0\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q0_scores]), np.mean(test_q0_scores), np.mean(test_q0_ecrr), np.mean(test_q0_worse), np.mean(test_q0_worse)))
        print("q1\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q1_scores]), np.mean(test_q1_scores), np.mean(test_q1_ecrr), np.mean([compute_efficiency(s, 2) for s in val_q1_scores]), np.mean(test_q1_worse)))
        print("q2\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q2_scores]), np.mean(test_q2_scores), np.mean(test_q2_ecrr), np.mean([compute_efficiency(s, 3) for s in val_q1_scores]), np.mean(test_q2_worse)))
        print("base\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_base_scores]), np.mean(test_base_scores), np.mean(test_base_ecrr), np.mean(test_base_efficiency), np.mean(test_base_worse)))
        print("moirl\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_moirl_scores]), np.mean(test_moirl_scores), np.mean(test_moirl_ecrr), np.mean(test_moirl_efficiency), np.mean(test_moirl_worse)))   
        print("oracle\tacc %.6f, avgmrr %.6f, ecrr %.6f, efficiency %.6f, worse decisions 0" % 
            (np.mean([1 if score == 1 else 0 for score in test_oracle_scores]), np.mean(test_oracle_scores), np.mean(test_oracle_ecrr), np.mean(test_oracle_efficiency)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, default = 'MSDialog')
    parser.add_argument('--topn', type = int, default = 1)
    parser.add_argument('--cv', type = str, default = '')
    parser.add_argument('--reranker_name', type = str, default = 'poly')
    parser.add_argument('--user_patience', type = int, default = 10)
    parser.add_argument('--cascade_p', type = float, default = 0.9)
    parser.add_argument('--reranker_return_length', type = int, default = 10)
    parser.add_argument('--n_policy', type = int, default = 1)
    parser.add_argument('--observation_dim', type = int, default = 768)
    parser.add_argument('--lr', type = float, default = 5e-5)
    parser.add_argument('--weight_decay', type = float, default = 1e-2)
    parser.add_argument('--n_action', type = int, default = 2)
    parser.add_argument('--cq_reward', type = float, default = 0.1)
    parser.add_argument('--train_iter', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--max_data_size', type = int, default = 10000)
    parser.add_argument('--test', type = float, default = 0.1)
    parser.add_argument('--checkpoint', type = str, default = '')
    parser.add_argument('--checkpoint_base', type = str, default = '')
    parser.add_argument('--checkpoint_moirl', type = str, default = '')
    parser.add_argument('--load_checkpoint', type = bool, default = False)

    args = parser.parse_args()
    main(args)
