from __future__ import print_function
import os
import torch 
import torch.nn as nn
import numpy as np
import wandb

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 

def train_model(model, scheduler, trainingLoader, optim, validationLoader, attributes, **kwargs):    
    save_path = kwargs.get('save_path')
    n_epochs = kwargs.get("n_epochs")
    data_window = kwargs.get("LSTM_split")
    
    # Kaydedilecek veriler
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    
    # Epoch döngüsü
    for epoch in range(n_epochs): 
        model.train()
        batch_losses = []
        train_percent_errors = []
        
        # Mini-batch döngüsü
        for batch_idx, data in enumerate(trainingLoader):
            optim.zero_grad()
            sig, stat_info, gt = data
            sig = sig.to(device) 
            stat_info = stat_info.to(device)
            gt = gt.to(device)
            
            predictions = model(sig, stat_info, data_window)
            
            # Kayıp hesapla ve optimizasyon yap
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, gt)
            batch_losses.append(loss.item())
            
            # Yüzdesel fark hesapla
            percent_errors = torch.abs((predictions - gt) / gt) * 100
            train_percent_errors.extend(percent_errors.detach().cpu().numpy())
            
            loss.backward()
            optim.step()
        
        # Eğitim kaybını ve doğruluğunu hesapla
        training_loss = np.mean(batch_losses) 
        training_losses.append(training_loss)
        
        training_accuracy_percentage = 100 - np.mean(train_percent_errors)
        training_error_percentage = np.mean(train_percent_errors)
        training_accuracies.append(training_accuracy_percentage)
        
        # Modeli doğrulama moduna al ve validation doğruluğu hesapla
        model.eval()
        val_losses = []
        val_percent_errors = []
        val_abs_diffs = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(validationLoader):
                sig, stat_info, gt = data
                sig = sig.to(device)
                stat_info = stat_info.to(device)
                gt = gt.to(device)
                
                predictions = model(sig, stat_info, data_window)
                loss = nn.MSELoss()(predictions, gt)
                val_losses.append(loss.item())
                
                # Yüzdesel fark hesapla
                percent_errors = torch.abs((predictions - gt) / gt) * 100
                val_percent_errors.extend(percent_errors.cpu().numpy())
                
                # Gerçek ve tahmin edilen değerlerin farkını hesapla
                abs_diffs = torch.abs(predictions - gt)
                val_abs_diffs.extend(abs_diffs.cpu().numpy())
        
        # Validation kaybı ve doğruluğunu hesapla
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)
        
        validation_accuracy_percentage = 100 - np.mean(val_percent_errors)
        validation_error_percentage = np.mean(val_percent_errors) 
        validation_accuracies.append(validation_accuracy_percentage)
        
        # Ortalama mutlak fark hesapla
        validation_abs_diff_mean = np.mean(val_abs_diffs)
        
        # Wandb loglaması
        wandb.log({
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_accuracy_percentage": training_accuracy_percentage,
            "training_error_percentage": training_error_percentage,
            "validation_accuracy_percentage": validation_accuracy_percentage,
            "validation_error_percentage": validation_error_percentage,
            "validation_absolute_difference": validation_abs_diff_mean,
            "epoch": epoch
        })
    
    # Modeli kaydet
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "finalModel_state_dict.pt"))
    torch.save(model, os.path.join(save_path, "finalModel.pt"))
    
    return training_losses, validation_losses