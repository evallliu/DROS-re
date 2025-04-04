import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import os
from utils.utility import pad_history,calculate_hit,extract_axis_1,parse_args
from collections import Counter
from utils.SASRecModules_ori import *

def evaluate(model, test_data, device, data_directory, seq_size, item_num, reward_click):
    topk=[10, 20, 50]

    eval_sessions=pd.read_pickle(os.path.join(data_directory, test_data))
    if model.__class__.__name__ == 'SHAN':
        eval_sessions['user_id'] = 1
    eval_ids = eval_sessions.session_id.unique() #eval_ids 是 session_id 的唯一值，表示每个评估会话的 ID。通常用于区分不同的用户或会话?
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated=0
    total_clicks=1.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]
    while evaluated<len(eval_ids):
        states, len_states, actions, rewards, user_ids = [], [], [], [], []
        for i in range(batch):
            id=eval_ids[evaluated]
            group=groups.get_group(id)
            history=[]
            for index, row in group.iterrows():
                state=list(history)
                state = [int(i) for i in state]
                len_states.append(seq_size if len(state)>=seq_size else 1 if len(state)==0 else len(state))
                state=pad_history(state,seq_size,item_num)
                states.append(state)
                action=row['item_id']
                try:
                    is_buy=row['t_read']
                except:
                    is_buy=row['time']
                reward = 1 if is_buy >0 else 0
                if is_buy>0:
                    total_purchase+=1.0
                else:
                    total_clicks+=1.0
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
                if model.__class__.__name__ == 'SHAN':
                    user_ids.append(row['user_id'])
            evaluated+=1
            if evaluated >= len(eval_ids):
                break

        states = np.array(states)
        states = torch.LongTensor(states)
        states = states.to(device)

        user_ids = np.array(user_ids)  # 将 user_ids 转换为 numpy 数组
        user_ids = torch.LongTensor(user_ids).to(device)  # 转换为 LongTensor，并移动到设备

        if model.__class__.__name__ == 'SHAN':
            prediction = model.forward_eval(states, user_ids)
        else:
            prediction = model.forward_eval(states, np.array(len_states))
        # print(prediction)
        prediction = prediction.cpu()
        prediction = prediction.detach().numpy()
        # print(prediction)
        # prediction=sess.run(GRUnet.output, feed_dict={GRUnet.inputs: states,GRUnet.len_state:len_states,GRUnet.keep_prob:1.0})
        sorted_list=np.argsort(prediction)
        calculate_hit(sorted_list,topk,actions,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)
    print('#############################################################')
    # logging.info('#############################################################')
    # print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    # logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    hr_list = []
    ndcg_list = []
    print('hr@{}\tndcg@{}\thr@{}\tndcg@{}\thr@{}\tndcg@{}'.format(topk[0], topk[0], topk[1], topk[1], topk[2], topk[2]))
    # logging.info('#############################################################')
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])

        if i == 1:
            hr_20 = hr_purchase

    print('{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    print('{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    # logging.info('{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    # logging.info('{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    print('#############################################################')
    # logging.info('#############################################################')

    return hr_20

def calcu_propensity_score(buffer, item_num):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)
    return ps

def train(args, item_num, optimizer, device, model, data_directory, seq_size,reward_click):
    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    if args.model_name == 'SHAN':
        train_data['user_id'] = 1

    ps = calcu_propensity_score(train_data, item_num)
    ps = torch.tensor(ps)
    ps = ps.to(device)

    total_step=0
    hr_max = 0
    best_epoch = 0

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)
    for i in range(args.epoch):
            for j in range(num_batches):
                batch = train_data.sample(n=args.batch_size).to_dict()
                seq = list(batch['seq'].values())
                len_seq = list(batch['len_seq'].values())
                target=list(batch['next'].values())

                target_neg = []
                for index in range(args.batch_size):
                    neg=np.random.randint(item_num)
                    while neg==target[index]:
                        neg = np.random.randint(item_num)
                    target_neg.append(neg)
                optimizer.zero_grad()
                seq = torch.LongTensor(seq)
                if args.model_name == 'SASRec' or args.model_name == 'BERT4Rec':
                    len_seq = torch.LongTensor(len_seq)
                target = torch.LongTensor(target)
                target_neg = torch.LongTensor(target_neg)
                seq = seq.to(device)
                target = target.to(device)
                if args.model_name == 'SASRec' or args.model_name == 'BERT4Rec':
                    len_seq = len_seq.to(device)
                target_neg = target_neg.to(device)
                
                if args.model_name == 'SHAN':
                    user = list(batch['user_id'].values())##
                    len_user = list(batch['user_id'].values())##
                    user = torch.LongTensor(user)
                    user = user.to(device)
                # if args.model_name == 'SHAN':
                    model_output = model.forward(seq, user)
                else:
                    model_output = model.forward(seq, len_seq)

                if args.loss_type == 'bpr':
                    model_output = F.elu(model_output) + 1
                # print('model_output:',model_output.size())

                target = target.view(args.batch_size, 1)
                target_neg = target_neg.view(args.batch_size, 1)
                # print('target&neg:',target.size(), target_neg.size())

                pos_scores = torch.gather(model_output, 1, target)
                neg_scores = torch.gather(model_output, 1, target_neg)

                pos_labels = torch.ones((args.batch_size, 1))
                neg_labels = torch.zeros((args.batch_size, 1))

                scores = torch.cat((pos_scores, neg_scores), 0)
                labels = torch.cat((pos_labels, neg_labels), 0)
                # print('labels:',labels.size())
                labels = labels.to(device)

                if args.loss_type == 'bce':
                    bce_loss = nn.BCEWithLogitsLoss()
                    loss = bce_loss(scores, labels)
                elif args.loss_type == 'bpr':
                    loss = -torch.log(1e-24 + torch.sigmoid(pos_scores - neg_scores)).mean()
                elif args.loss_type == 'mse':
                    mse_loss = nn.MSELoss()
                    loss = mse_loss(scores, labels)
                else:
                    print('Got a wrong loss type.')

                pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
                pos_scores_dro = torch.squeeze(pos_scores_dro)
                pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
                pos_loss_dro = torch.squeeze(pos_loss_dro)

                inner_dro = torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1) - torch.exp((pos_scores_dro / args.beta)) + torch.exp((pos_loss_dro / args.beta))

                # A = torch.sum(torch.exp(torch.mul(model_output * model_output, ps)), 1)
                # B = torch.exp(pos_scores_dro)
                # C = torch.exp(pos_loss_dro) 
                # print(A.shape, B.shape, C.shape)

                loss_dro = torch.log(inner_dro + 1e-24)
                if args.alpha == 0.0:
                    loss_all = loss
                else:
                    loss_all = loss + args.alpha * torch.mean(loss_dro)
                loss_all.backward()
                optimizer.step()

                if True:

                    total_step+=1
                    if total_step % 200 == 0:
                        print("the loss in %dth step is: %f" % (total_step, loss_all))
                        # logging.info("the loss in %dth step is: %f" % (total_step, loss_all))

                    if total_step % 2000 == 0:
                            # print('VAL:')
                            # logging.info('VAL:')
                            # hr_20 = evaluate(model, 'val_sessions_pos.df', device)
                            print('VAL PHRASE:')
                            # logging.info('VAL PHRASE:')
                            hr_20 = evaluate(model, 'val_sessions.df', device, data_directory, seq_size, item_num, reward_click)
                            print('TEST PHRASE:')
                            # logging.info('TEST PHRASE:')
                            _ = evaluate(model, 'test_sessions.df', device, data_directory, seq_size, item_num, reward_click)
                            # print('TEST PHRASE3:')
                            # logging.info('TEST PHRASE3:')
                            # _ = evaluate(model, 'test_sessions3_pos.df', device)

                            if hr_20 > hr_max:

                                hr_max = hr_20
                                best_epoch = total_step
                            
                            print('BEST EPOCH:{}'.format(best_epoch))
                            # logging.info('BEST EPOCH:{}'.format(best_epoch))
                    if total_step > 2000:
                        break
