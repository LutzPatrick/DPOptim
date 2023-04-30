import logging
import torch
import copy

logger = logging.getLogger('core')
logger.setLevel(logging.INFO)
#logger.addHandler(logging.StreamHandler(sys.stdout))

def test(data, model, mode, epoch, metrics, device):
    for batch, target in data[mode]:
        pred = model(batch.to(device))
        for metric_fn in metrics.metric_fns:
            metrics[metric_fn, mode, epoch] += metric_fn(pred.detach(), target.to(device))*len(batch)
    for metric_fn in metrics.metric_fns:
        metrics[metric_fn, mode, epoch] /= len(data[mode].dataset)

def train(data, model, optimizer, loss_fn, epochs, metrics, device, ponc=False):
    
    with torch.no_grad():
        test(data, model, 'train', 0, metrics, device)
        test(data, model, 'val', 0, metrics, device)
        test(data, model, 'test', 0, metrics, device)
    logger.info(f'[{0}/{epochs}] loss train {metrics[loss_fn, "train", 0]:.5}, '
            f'val {metrics[loss_fn, "val", 0]:.5}, test {metrics[loss_fn, "test", 0]:.5}')

    for epoch in range(1, epochs+1):

        if ponc:
            train_one_epoch_ponc(data, model, loss_fn, optimizer, metrics, epoch, device)
        else:
            train_one_epoch_sgd(data, model, loss_fn, optimizer, metrics, epoch, device)

        with torch.no_grad():
            test(data, model, 'val', epoch, metrics, device)
            test(data, model, 'test', epoch, metrics, device)

        logger.info(f'[{epoch}/{epochs}] loss train {metrics[loss_fn, "train", epoch]:.5}, '
            f'val {metrics[loss_fn, "val", epoch]:.5}, test {metrics[loss_fn, "test", epoch]:.5}')

def train_one_epoch_sgd(data, model, loss_fn, optimizer, metrics, epoch, device):

    for batch, target in data['train']:
        batch, target = batch.to(device), target.to(device)
        pred = model(batch)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for metric_fn in metrics.metric_fns:
            metrics[metric_fn, 'train', epoch] += metric_fn(pred.detach(), target)*len(batch)

    for metric_fn in metrics.metric_fns:
        metrics[metric_fn, 'train', epoch] /= len(data['train'].dataset)

def train_one_epoch_ponc(data, model, loss_fn, optimizer, metrics, epoch, device):

    model_prev = copy.deepcopy(model)
    for i, (batch, target) in enumerate(data['train']):
        batch, target = batch.to(device), target.to(device)

        if not i % 2:
            # checkpoint gradient
            pred = model(batch)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        else:
            # consider each element in batch individually
            for i in range(len(batch)):
                batchi = batch[i, ...][None, ...]
                targeti = target[i, ...][None, ...]
                # current weight
                pred = model(batchi)
                loss = loss_fn(pred, targeti)
                optimizer.zero_grad()
                loss.backward()
                # previous weight
                pred2 = model_prev(batchi)
                loss2 = loss_fn(pred2, targeti)
                model_prev.zero_grad()
                loss2.backward()
                model_prev.load_state_dict(model.state_dict())
                
                # update
                gradient_difference(model, model_prev)
                optimizer.step()


                model_prev.load_state_dict(model.state_dict())

            


        pred = model(batch)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for metric_fn in metrics.metric_fns:
            metrics[metric_fn, 'train', epoch] += metric_fn(pred.detach(), target)*len(batch)

    for metric_fn in metrics.metric_fns:
        metrics[metric_fn, 'train', epoch] /= len(data['train'].dataset)

def gradient_difference(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p1.grad -= p2.grad
        