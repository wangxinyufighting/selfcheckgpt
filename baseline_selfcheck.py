import json
import torch
import spacy
import time
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckNgram, SelfCheckBERTScore, SelfCheckMQAG
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import accuracy_score,\
    classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, precision_recall_curve, auc

# dataset = 'qa'
# og_smaples = './data/20_samples.json'
# og_passage = './data/data_llama2_7b_chat_multi_answer_all_one_answer_beam5_dosample_false.json'
# file = './data/data_llama2_7b_chat_multi_answer_all_one_answer_beam5_dosample_false_with_label.json'
# test_qids = np.load('./test_qids.npy').tolist()

nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")



# key = 'nli'
# key = 'unigram'
key = 'bertscore'
# key = 'qa'

count = 0
all_time = 0
selfcheck = None

parent_path = '/home/wxy/models/{}'

if key == 'bertscore':
    selfcheck = SelfCheckBERTScore(rescale_with_baseline=True)
elif key == 'nli':
    selfcheck = SelfCheckNLI(device=device, nli_model=parent_path.format('deberta-v3-large-mnli')) # set device to 'cuda' if GPU is available
elif key == 'unigram':
    selfcheck = SelfCheckNgram(n=1) 
elif key == 'qa':
    selfcheck = SelfCheckMQAG(device=device
                              , g1_model=parent_path.format('t5-large-generation-squad-QuestionAnswer')
                              , g2_model=parent_path.format('t5-large-generation-race-QuestionAnswer')
                              , answering_model=parent_path.format('longformer-large-4096-answering-race')
                              , answerability_model=parent_path.format('longformer-large-4096-answerable-squad2')
                              )

# model_names = ['llama2-7b-chat-hf']
model_names = ['llama2-7b-chat-hf', 'Mistral-7B-Instruct-v0.2', 'vicuna-7b', 'vicuna-13b', 'vicuna-33b']
# model_names = ['vicuna-33b']
# model_name = 'vicuna-7b'
# model_name = 'llama2-7b-chat-hf'
for model_name in model_names:
    true_list = []
    mean_logits_list = []
    max_logits_list = []
    pred_list = []
    every_time_list = []
    count = 0
    with open(f'./dataset/gsm8k_{model_name}_20_samples_0shot_test.json', 'r') as f:
    # with open(f'./dataset/test.json', 'r') as f:
        for line in tqdm(f.readlines()):
            count += 1
            d = json.loads(line.strip())
            samples = d['samples']
            label = d['label']
            passage = d['predict']
            
            sentences = [sent.text.strip() for sent in nlp(passage).sents] # spacy sentence tokenization
            
            sentences_new = []
            for i in sentences:
                if i:
                    sentences_new.append(i)
                    
            samples_new = []
            for i in samples:
                if i and i.replace('\n', ''):
                    samples_new.append(i)

            if sentences_new and samples_new:
                true_list.append(label)

                starttime = time.time()
                if isinstance(selfcheck, SelfCheckBERTScore) or isinstance(selfcheck, SelfCheckNLI):
                    # try:
                    sent_scores = selfcheck.predict(
                        sentences = sentences_new,                          # list of sentences
                        sampled_passages = samples_new, # list of sampled passages
                    )
                    mean_logits_list.append(np.mean(sent_scores))
                    max_logits_list.append(np.max(sent_scores))
                    # except:
                    #     logits_list.append(float(random.random()))
                elif isinstance(selfcheck, SelfCheckMQAG):
                    # try:
                    sent_scores = selfcheck.predict(
                        sentences = sentences_new,               # list of sentences
                        passage = passage,                   # passage (before sentence-split)
                        sampled_passages = samples_new, # list of sampled passages
                        num_questions_per_sent = 5,          # number of questions to be drawn  
                        scoring_method = 'bayes_with_alpha', # options = 'counting', 'bayes', 'bayes_with_alpha'
                        beta1 = 0.8, beta2 = 0.8,            # additional params depending on scoring_method
                    )
                    # print(sent_scores)

                    mean_ = np.mean([i for i in sent_scores if not np.isnan(i)])
                    mean_logits_list.append(mean_)
                    max_ = np.max([i for i in sent_scores if not np.isnan(i)])
                    max_logits_list.append(max_) 
                    # except:
                    #     logits_list.append(float(random.random())) 
                else:
                    # try:
                    sent_scores = selfcheck.predict(
                            sentences = sentences_new,   
                            passage = passage,
                            sampled_passages = samples_new,
                        )
                    max_ = np.max(sent_scores['sent_level']['max_neg_logprob'])
                    mean_ = np.mean(sent_scores['sent_level']['max_neg_logprob'])
                    mean_logits_list.append(max_)
                    max_logits_list.append(mean_)
                    # except:
                    #     logits_list.append(float(random.randint(1,100)))

                endtime = time.time()
                # all_time += (endtime - starttime)
                every_time_list.append(endtime - starttime)
                # if count in [1, 10, 50, 100, 200, 500, 1000, 2000]:
                #     print(f'case num:{count}, time:{all_time}s, ')
    count = 0       
    np.save(f'./dataset/result/max_logits_list_{key}_{model_name}.npy', max_logits_list)
    np.save(f'./dataset/result/mean_logits_list_{key}_{model_name}.npy', mean_logits_list)
    np.save(f'./dataset/result/true_{key}_{model_name}.npy', true_list)
    np.save(f'./dataset/result/time_cost_{key}_{model_name}.npy', every_time_list)

    max_score = np.load(f'./dataset/result/max_logits_list_{key}_{model_name}.npy')
    mean_score = np.load(f'./dataset/result/mean_logits_list_{key}_{model_name}.npy')
    labels = np.load(f'./dataset/result/true_{key}_{model_name}.npy')

    auroc = roc_auc_score(labels,mean_score)
    precision, recall, _ = precision_recall_curve(labels,mean_score)

    print('mean', model_name, key, auroc, auc(recall, precision))

    auroc = roc_auc_score(labels,max_score)
    precision, recall, _ = precision_recall_curve(labels,max_score)

    print('max', model_name, key, auroc, auc(recall, precision))