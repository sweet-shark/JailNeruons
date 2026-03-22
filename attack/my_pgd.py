import torch

def normalize(image, mean, std):
    num_channel = len(image)
    mean = torch.tensor(mean, device=image.device)
    std = torch.tensor(std, device=image.device)
    mean = mean.unsqueeze(1).unsqueeze(1) # shape = 3, 1, 1
    std = std.unsqueeze(1).unsqueeze(1)
    for _ in range(num_channel-3):
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    denormalize_image = (image - mean)/std
    return denormalize_image


def denormalize(image, mean, std):
    num_channel = len(image)
    mean = torch.tensor(mean, device=image.device)
    std = torch.tensor(std, device=image.device)
    mean = mean.unsqueeze(1).unsqueeze(1) # shape = 3, 1, 1
    std = std.unsqueeze(1).unsqueeze(1)
    for _ in range(num_channel-3):
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    denormalize_image = image * std + mean
    return denormalize_image

def my_pgd(model, image, alpha = 0.003, num_epoch = 300, target=True):
    noise = torch.zeros(image.shape).to(image.device)
    for j in range(num_epoch):
        for p in range(1):
            image.requires_grad = True
            inp = []

            adv_image = image + noise 

            inp.append(adv_image)
            inp.append(p)
            loss = model(inp)
            # Update adversarial images
            # print(adv_image.requires_grad)
            # raise
            grad = torch.autograd.grad(
                loss, adv_image, retain_graph=False, create_graph=False
            )[0]
            # raise
            # cost_step += cost.clone().detach()

            adv_image = adv_image.detach() + alpha * grad.sign()
            delta = adv_image - image
            adv_image = torch.clamp(image + delta, min=0, max=1).detach()
            noise = adv_image - image
            
    return adv_image
        
    