import logging
import torch
import copy
from opacus import GradSampleModule
from models import get_model

logger = logging.getLogger('core')
logger.setLevel(logging.INFO)
#logger.addHandler(logging.StreamHandler(sys.stdout))

def test(data, model, mode, epoch, metrics, device):
    data = data[mode]
    if type(data) == tuple:
        data = data[0]

    for batch, target in data:
        pred = model(batch.to(device))
        for metric_fn in metrics.metric_fns:
            metrics[metric_fn, mode, epoch] += metric_fn(pred.detach(), target.to(device))*len(batch)
    for metric_fn in metrics.metric_fns:
        metrics[metric_fn, mode, epoch] /= len(data.dataset)

def train(data, model, optimizer, loss_fn, epochs, metrics, device, dataset, ponc=False):
    
    with torch.no_grad():
        test(data, model, 'train', 0, metrics, device)
        test(data, model, 'test', 0, metrics, device)
    logger.info(f'[{0}/{epochs}] train loss {metrics[loss_fn, "train", 0]:.5} acc {metrics[metrics.metric_fns[1], "train", 0]:.5}, '
            f'test loss {metrics[loss_fn, "test", 0]:.5} acc {metrics[metrics.metric_fns[1], "train", 0]:.5}')

    for epoch in range(1, epochs+1):

        if ponc:
            train_one_epoch_ponc(data, model, loss_fn, optimizer, dataset, device)
        else:
            train_one_epoch_sgd(data, model, loss_fn, optimizer, device)

        with torch.no_grad():
            test(data, model, 'train', epoch, metrics, device)
            test(data, model, 'test', epoch, metrics, device)

        logger.info(f'[{epoch}/{epochs}] train loss {metrics[loss_fn, "train", epoch]:.5} acc {metrics[metrics.metric_fns[1], "train", epoch]:.5}, '
            f'test loss {metrics[loss_fn, "test", epoch]:.5} acc {metrics[metrics.metric_fns[1], "train", epoch]:.5}')

def train_one_epoch_sgd(data, model, loss_fn, optimizer, device):

    for batch, target in data['train']:
        batch, target = batch.to(device), target.to(device)
        pred = model(batch)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_one_epoch_ponc(data, model, loss_fn, optimizer, dataset, device):

    train_cp, train_diff = data['train']
    train_cp, train_diff = iter(train_cp), iter(train_diff)
    grad_checkpoint, ponc_descent = optimizer

    model_prev = get_model(dataset, device)
    if isinstance(model, GradSampleModule):
        model_prev = GradSampleModule(model_prev)
        
    for _ in range(len(train_cp)):
        batch_cp, target_cp = next(train_cp)
        batch_cp, target_cp = batch_cp.to(device), target_cp.to(device)
        
        # checkpoint gradient
        pred = model(batch_cp)
        loss = loss_fn(pred, target_cp)
        grad_checkpoint.zero_grad()
        loss.backward()
        model_prev.load_state_dict(model.state_dict())
        grad_checkpoint.step()
        logger.info(f'cp loss {loss}')
        
        for _ in range(len(batch_cp-1)):
            batch_diff, target_diff = next(train_diff)
            batch_diff, target_diff = batch_diff.to(device), target_diff.to(device)
            # current weight
            pred = model(batch_diff)
            loss = loss_fn(pred, target_diff)
            ponc_descent.zero_grad()
            loss.backward()
            # previous weight
            pred2 = model_prev(batch_diff)
            loss2 = loss_fn(pred2, target_diff)
            model_prev.zero_grad()
            loss2.backward()
            # update
            gradient_difference(model, model_prev)
            model_prev.load_state_dict(model.state_dict())
            ponc_descent.step()

def gradient_difference(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p1.grad -= p2.grad
        