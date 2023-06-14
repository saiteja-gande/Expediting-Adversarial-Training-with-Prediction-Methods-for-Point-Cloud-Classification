import torch

def fgsm_attack(model, criterion, point, labels, eps, x = False, m = True) :
    point.requires_grad = True
    if m == True:
        outputs, __, __ = model(point.transpose(1,2), x) #Fgsm Attack to PointNet Network
    else :
        outputs = model(point.permute(0,2,1)) #Fgsm Attack to DGCNN Network
    model.zero_grad()
    cost = criterion(outputs, labels.long())
    cost.backward()
    attack_data = eps*point.grad.sign()
    return attack_data

def pgd_linf(model, x, y, loss_fn, num_steps, step_size, eps, xs = False , m = True):
    delta = torch.zeros_like(x, requires_grad=True)
    for i in range(num_steps):
        if m == True :
            prediction, __, __ = model((x+delta).transpose(1,2), xs) #Pgd Attack to PointNet Network
        else :
           prediction = model((x+delta).permute(0,2,1)) #Pgd Attack to DGCNN Network
        loss = loss_fn(prediction,y)
        loss.backward()
        delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-eps,eps)
        delta.grad.zero_()
    return delta.detach()
