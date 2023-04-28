import logging
import torch
import sys

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

def train(data, model, optimizer, loss_fn, epochs, metrics, device):
    
    with torch.no_grad():
        test(data, model, 'train', 0, metrics, device)
        test(data, model, 'val', 0, metrics, device)
        test(data, model, 'test', 0, metrics, device)
    logger.info(f'[{0}/{epochs}] loss train {metrics[loss_fn, "train", 0]:.5}, '
            f'val {metrics[loss_fn, "val", 0]:.5}, test {metrics[loss_fn, "test", 0]:.5}')

    for epoch in range(1, epochs+1):

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

        with torch.no_grad():
            test(data, model, 'val', epoch, metrics, device)
            test(data, model, 'test', epoch, metrics, device)

        logger.info(f'[{epoch}/{epochs}] loss train {metrics[loss_fn, "train", epoch]:.5}, '
            f'val {metrics[loss_fn, "val", epoch]:.5}, test {metrics[loss_fn, "test", epoch]:.5}')


        