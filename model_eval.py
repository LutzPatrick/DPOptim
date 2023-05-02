import torch
import torch.nn as nn
import yaml
import logging
import pickle
import warnings
import argparse
from utils import get_optimizer, accuracy, hp_from_file, make_dp, get_dataloader
from models import get_model
from datasets import Metrics
from core import train


warnings.simplefilter("ignore")
logger = logging.getLogger('eval')
logger.setLevel(logging.INFO)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MNIST'])
    parser.add_argument('--optim_name', choices=['SGD', 'DPSGD', 'ONC', 'PONC'])
    parser.add_argument('--n_threads', default=0)
    parser.add_argument('--runs', type=int)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--delta', type=float)
    args = parser.parse_args()

    # load HP
    with open(f'./HP/{args.optim_name}/{args.dataset}.yaml', 'r') as file:
        hp = hp_from_file(yaml.safe_load(file))

    # print some info
    logger.info(f'evaluate {args.optim_name} on {args.dataset}')
    logger.info(f'hp setting {hp}')
    
    # load data. note: for now, data['val'] = data['test']
    data = get_dataloader(args, hp)

    # define learning objects shared over all runs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss()
    metrics = Metrics([loss_fn, accuracy], args.runs, hp['epochs'], device)

    for run in range(args.runs):
        logger.info(f'##### RUN {run+1} / {args.runs} #####')
        # get model, optimizer and make DP
        model = get_model(args.dataset, device)
        optimizer = get_optimizer(args.optim_name, model, hp)
        privacy_engine, model, optimizer, data['train'] = make_dp(model, optimizer, data['train'], args, hp, logger)
        # train model, stats are saved in metrics object
        ponc = args.optim_name in ['ONC', 'PONC']
        train(data, model, optimizer, loss_fn, hp['epochs'], metrics, device, args.dataset, ponc=ponc)
        # increment run counter in metrics object
        metrics.increment_run()
        if args.optim_name in ['DPSGD']:
            logger.info(f'privacy budget spent {(privacy_engine.get_epsilon(args.delta), args.delta)}')
        if args.optim_name in ['PONC']:
            privacy_engine1, privacy_engine2 = privacy_engine
            logger.info(f'privacy budget spent on checkpoint {(privacy_engine1.get_epsilon(args.delta), args.delta/2)}')
            logger.info(f'privacy budget spent on ponc descent {(privacy_engine2.get_epsilon(args.delta), args.delta/2)}')

    # copy metrics to cpu
    metrics.to('cpu')

    # print some metrin information
    train_acc, val_acc, test_acc = metrics.get_summary(accuracy)
    logger.info(f'acc train {train_acc}\n val {val_acc}\n test {test_acc}')

    # save metrics
    with open(f'./runs/{args.optim_name}_{args.dataset}.pkl', 'wb') as file:
        pickle.dump(metrics, file)
