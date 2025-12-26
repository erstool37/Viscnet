def MAPEflowcalculator(pred, target, descaler, method, path):
    """
    for flow model, prediction comes in gaussian distribution values, and must have mean of 0 and std of 1 batchwise
    
    """
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    
    pred_den = descaler(pred[:,0], "density", path).unsqueeze(-1)
    pred_dynvisc = descaler(pred[:,1], "dynamic_viscosity", path).unsqueeze(-1)
    pred_surfT = descaler(pred[:,2], "surface_tension", path).unsqueeze(-1)

    target_den = descaler(target[:,0], "density", path).unsqueeze(-1)
    target_dynvisc = descaler(target[:,1], "dynamic_viscosity", path).unsqueeze(-1)
    target_surfT = descaler(target[:,2], "surface_tension", path).unsqueeze(-1)

    loss_mape_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
    loss_mape_dynvisc = torch.mean((torch.abs(pred_dynvisc - target_dynvisc) / target_dynvisc)).unsqueeze(-1)
    loss_mape_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)

    z_mean = pred.mean(dim=0)
    z_std = pred.std(dim=0)

    wandb.log({
        f"MAPE {method} den %" : loss_mape_den * 100,
        f"MAPE {method} dynvisc %" : loss_mape_dynvisc * 100,
        f"MAPE {method} surfT %" : loss_mape_surfT * 100,
        f"MAPE {method} den answer" : target_dynvisc.squeeze().tolist(),
        f"MAPE {method} dynvisc answer" : target_dynvisc.squeeze().tolist(),
        f"MAPE {method} surfT answer" : pred_dynvisc.squeeze().tolist(),

        "z_mean_den": z_mean[0].item(),
        "z_mean_visc": z_mean[1].item(),
        "z_mean_surf": z_mean[2].item(),
        "z_std_den": z_std[0].item(),
        "z_std_visc": z_std[1].item(),
        "z_std_surf": z_std[2].item()
        })

def MAPEflowcalculator(pred, target, descaler, method, path):
    """
    for flow model, prediction comes in gaussian distribution values, and must have mean of 0 and std of 1 batchwise
    
    """
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    
    pred_den = descaler(pred[:,0], "density", path).unsqueeze(-1)
    pred_dynvisc = descaler(pred[:,1], "dynamic_viscosity", path).unsqueeze(-1)
    pred_surfT = descaler(pred[:,2], "surface_tension", path).unsqueeze(-1)

    target_den = descaler(target[:,0], "density", path).unsqueeze(-1)
    target_dynvisc = descaler(target[:,1], "dynamic_viscosity", path).unsqueeze(-1)
    target_surfT = descaler(target[:,2], "surface_tension", path).unsqueeze(-1)

    loss_mape_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
    loss_mape_dynvisc = torch.mean((torch.abs(pred_dynvisc - target_dynvisc) / target_dynvisc)).unsqueeze(-1)
    loss_mape_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)

    z_mean = pred.mean(dim=0)
    z_std = pred.std(dim=0)

    wandb.log({
        f"MAPE {method} den %" : loss_mape_den * 100,
        f"MAPE {method} dynvisc %" : loss_mape_dynvisc * 100,
        f"MAPE {method} surfT %" : loss_mape_surfT * 100,
        f"MAPE {method} den answer" : target_dynvisc.squeeze().tolist(),
        f"MAPE {method} dynvisc answer" : target_dynvisc.squeeze().tolist(),
        f"MAPE {method} surfT answer" : pred_dynvisc.squeeze().tolist(),

        "z_mean_den": z_mean[0].item(),
        "z_mean_visc": z_mean[1].item(),
        "z_mean_surf": z_mean[2].item(),
        "z_std_den": z_std[0].item(),
        "z_std_visc": z_std[1].item(),
        "z_std_surf": z_std[2].item()
        })