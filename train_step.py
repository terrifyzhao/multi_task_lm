import torch
from annlp import get_device

device = get_device()


def base_step(batch, optim, model, amp):
    all_loss = 0
    for index in range(len(batch)):
        optim.zero_grad()
        output = model(batch[index]['input_ids'].to(device),
                       batch[index]['attention_mask'].to(device),
                       labels=batch[index]['labels'].to(device),
                       task_id=index)
        loss = output.loss
        all_loss += loss.item()
        if torch.cuda.is_available():
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optim.step()
    return all_loss / len(batch)


def weight_step(batch, optim, model, amp):
    all_loss = 0
    for index in range(len(batch)):
        optim.zero_grad()
        output = model(batch[index]['input_ids'].to(device),
                       batch[index]['attention_mask'].to(device),
                       labels=batch[index]['labels'].to(device),
                       task_id=index)
        loss = output.loss

        loss = torch.log(loss)

        all_loss += loss.item()
        if torch.cuda.is_available():
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optim.step()
    return all_loss / len(batch)


def grad_accumulation_step(batch, optim, model, amp, with_log=False):
    all_loss = None
    optim.zero_grad()
    for index in range(len(batch)):
        output = model(batch[index]['input_ids'].to(device),
                       batch[index]['attention_mask'].to(device),
                       labels=batch[index]['labels'].to(device),
                       task_id=index)
        loss = output.loss
        if all_loss:
            if with_log:
                all_loss += torch.log(loss)
            else:
                all_loss += loss
        else:
            if with_log:
                all_loss = torch.log(loss)
            else:
                all_loss = loss

        if torch.cuda.is_available():
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    optim.step()

    return all_loss.item() / len(batch)
