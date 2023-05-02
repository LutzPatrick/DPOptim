from opacus import PrivacyEngine
import torch
from optimizer import ONC_Descent
from datasets import load_data

def get_optimizer(optim_name, model, hp):
    if optim_name in ['SGD', 'DPSGD']:
        return torch.optim.SGD(model.parameters(), lr=hp['lr'], momentum=hp['momentum'])
    if optim_name in ['ONC', 'PONC']:
        grad_checkpoint = ONC_Descent(model.parameters(), lr=hp['lr'], D=hp['D'], batch_size=hp['batch_size'])
        ponc_descent = ONC_Descent(model.parameters(), lr=hp['lr'], D=hp['D'], batch_size=hp['batch_size'], grad_checkpoint=grad_checkpoint)
        return (grad_checkpoint, ponc_descent)
    raise ValueError()

def get_dataloader(args, hp):
    if args.optim_name in ['SGD', 'DPSGD']:
        data = load_data(args.dataset)
        for key, val in data.items():
            data[key] = torch.utils.data.DataLoader(val, batch_size=hp['batch_size'], shuffle=True, num_workers=args.n_threads)

    elif args.optim_name in ['ONC', 'PONC']:
        data = load_data(args.dataset)
        #data['val'] = torch.utils.data.DataLoader(data['val'], batch_size=hp['batch_size'], shuffle=True, num_workers=args.n_threads)
        data['test'] = torch.utils.data.DataLoader(data['test'], batch_size=hp['batch_size'], shuffle=True, num_workers=args.n_threads)
        
        train_cp, train_diff = torch.utils.data.random_split(data['train'], [.5, .5])
        train_cp = torch.utils.data.DataLoader(train_cp, batch_size=hp['batch_size'], shuffle=True, num_workers=args.n_threads)
        train_diff = torch.utils.data.DataLoader(train_diff, batch_size=1, shuffle=True, num_workers=args.n_threads)
        data['train'] = (train_cp, train_diff)

    else:
        ValueError()

    return data

def accuracy(x, y):
    return torch.mean((torch.argmax(x, dim=1) == y).double())

def hp_from_file(hp):
    hp['batch_size'] = 2**hp['batch_size_base']
    if 'momentum_base' in hp.keys():
        hp['momentum'] = hp['momentum_base']/float(10)
    return hp

def make_dp(model, optimizer, data, args, hp, logger):

    privacy_engine = None

    if args.optim_name in ['DPSGD']:
        privacy_engine = PrivacyEngine()
        model, optimizer, data = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data,
            max_grad_norm=hp['clip'],
            epochs=hp['epochs'],
            target_epsilon=args.epsilon,
            target_delta=args.delta,
        )
        logger.info(f'DP training using sigma={optimizer.noise_multiplier} and clip={hp["clip"]}')

    if args.optim_name in ['PONC']:

        grad_checkpoint, ponc_descent = optimizer
        train_cp, train_diff = data

        privacy_engine1 = PrivacyEngine()
        model, grad_checkpoint, data = privacy_engine1.make_private_with_epsilon(
            module=model,
            optimizer=grad_checkpoint,
            data_loader=train_cp,
            max_grad_norm=hp['clip'],
            epochs=hp['epochs'],
            target_epsilon=args.epsilon*hp['eps_frac_cp'],
            target_delta=args.delta,
        )
        logger.info(f'DP training using sigma={grad_checkpoint.noise_multiplier} and clip={hp["clip"]} for grad_checkpoint')

        privacy_engine2 = PrivacyEngine()
        model, ponc_descent, data = privacy_engine2.make_private_with_epsilon(
            module=model,
            optimizer=ponc_descent,
            data_loader=train_diff,
            max_grad_norm=hp['D'],
            epochs=hp['epochs'],
            target_epsilon=args.epsilon*(1-hp['eps_frac_cp']),
            target_delta=args.delta,
        )
        logger.info(f'DP training using sigma={ponc_descent.noise_multiplier} and clip={hp["D"]} for ponc_descent')

        privacy_engine = (privacy_engine1, privacy_engine2)
        optimizer = (grad_checkpoint, ponc_descent)
        data = (train_cp, train_diff)

    return privacy_engine, model, optimizer, data