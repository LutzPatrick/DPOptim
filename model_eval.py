import torch
import torch.nn as nn
import yaml
import logging
import pickle
import warnings
import argparse
from opacus import PrivacyEngine
from utils import get_optimizer, accuracy, hp_from_file
from models import get_model
from datasets import load_data, Metrics
from core import train


warnings.simplefilter("ignore")
logger = logging.getLogger('eval')
logger.setLevel(logging.INFO)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MNIST'])
    parser.add_argument('--optim_name', choices=['SGD', 'DPSGD'])
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
    data = load_data(args.dataset)
    for key, val in data.items():
        data[key] = torch.utils.data.DataLoader(val, batch_size=hp['batch_size'], shuffle=True, num_workers=args.n_threads)

    # define learning objects shared over all runs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss()
    metrics = Metrics([loss_fn, accuracy], args.runs, hp['epochs'], device)

    for run in range(args.runs):
        logger.info(f'##### RUN {run+1} / {args.runs} #####')
        # get model, optimizer and make DP
        model = get_model(args.dataset, device)
        optimizer = get_optimizer(args.optim_name, model, hp)
        if args.optim_name in ['DPSGD']:
            privacy_engine = PrivacyEngine()
            model, optimizer, data['train'] = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data['train'],
                #noise_multiplier=args.noise_multiplier,
                max_grad_norm=hp['clip'],
                epochs=hp['epochs'],
                target_epsilon=args.epsilon,
                target_delta=args.delta,
            )
            logger.info(f'DP training using sigma={optimizer.noise_multiplier} and clip={hp["clip"]}')
        # train model, stats are saved in metrics object
        train(data, model, optimizer, loss_fn, 40, metrics, device, ponc=True)
        # increment run counter in metrics object
        metrics.increment_run()
        if args.optim_name in ['DPSGD']:
            logger.info(f'privacy budget spent {(privacy_engine.get_epsilon(args.delta), args.delta)}')

    # copy metrics to cpu
    metrics.to('cpu')

    # print some metrin information
    train_acc, val_acc, test_acc = metrics.get_summary(accuracy)
    logger.info(f'acc train {train_acc}\n val {val_acc}\n test {test_acc}')

    # save metrics
    with open(f'./runs/{args.optim_name}_{args.dataset}.pkl', 'wb') as file:
        pickle.dump(metrics, file)
