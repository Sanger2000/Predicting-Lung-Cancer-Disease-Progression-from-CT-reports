import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import datetime
import numpy as np

def train_model(train_data, dev_data, model, args):


    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr)

    model.train()

    for epoch in range(1, args.epochs+1):

        print("-------------\nEpoch {}:\n".format(epoch))


        loss = run_epoch(train_data[0], train_data[1], train_data[2], True, model, optimizer, args)

        print('Train MSE loss: {:.6f}'.format( loss))

        print()

        val_loss = run_epoch(dev_data, False, model, optimizer, args)
        print('Val MSE loss: {:.6f}'.format( val_loss))

        # Save model
        torch.save(model, args.save_path)

def run_epoch(data, is_training, model, optimizer, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    losses = []

    if is_training:
        model.train()
    else:
        model.eval()

    for batch in tqdm(data_loader):

        x, y, z, labs = autograd.Variable(batch['baseX']), autograd.Variable(batch['progX'], autograd.Variable(batch['text']). autograd.Variable(batch['labels']))
        if args.cuda:
            x, y, z, labs = x.cuda(), y.cuda(), z.cuda(), labs.cuda()

        if is_training:
            optimizer.zero_grad()

        out = model(x, y, z)
        loss = F.mse_loss(out, labs.float())


        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss
