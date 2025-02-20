from __future__ import print_function
import torch 
import torch.nn as nn
from math import radians, cos, sin, asin, sqrt
import numpy as np
import wandb

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 


def inference_experiment(loader, attributes,signal_height, signal_width, model, phase, **kwargs):     

    metric_dist = []
    Vs30_diff =[]
    Vs30_percentage=[]
    column_wise_errors = []
    gt_select = kwargs.get("gt_select")
    data_window = kwargs.get("LSTM_split")
    true_values = [[] for ii in range(len(gt_select))]
    predictions = [[] for ii in range(len(gt_select))]
    data = [[] for ii in range(len(gt_select))]
    column_wise_errors = [[] for ii in range(len(gt_select))]
    model.eval()
    
    with torch.no_grad(): 
        losses = []
        for batch_idx, data_ in enumerate(loader):
            
            sig, stat_info, gt = data_ #sig = batchSize x 3 x accelPoints x 1
            sig = sig.to(device) 
            stat_info = stat_info.to(device) #☺batchSize x 3
            gt=gt.to(device)  #☺batchSize x varyingLength
            
            outputs=model(sig, stat_info,data_window) #same shape as gt
     
            loss=nn.MSELoss() #MSE loss object
            output = loss(outputs, gt) # L2 norm batch ortalaması (reduction default olarak mean)
            losses.append(output.item()) #accumulate the losses of each batch in losses, her batch için bir loss değeri kaydediliyor 
            
            
            # Calculate test losses 
            for i in range(len(gt[0])): # Per column of gt
                for mb in range(len(gt)): # Per elements in a  --Z loop iterates for 64 times if batchSize is 64                    
                    # Record Reversed Values
                    if kwargs.get("gtnorm"):
                        true_val = gt[mb][i] * attributes['trStds'][i] + attributes['trMeans'][i]
                        predicted_val = outputs[mb][i] * attributes['trStds'][i] + attributes['trMeans'][i]
                    else:
                        true_val = gt[mb][i]
                        predicted_val = outputs[mb][i]
                    
                    ######################
                    # Add stat_info back to relative distance differences
                    if kwargs.get("gt_select")[i] == 'epiLAT':
                        true_val = true_val + stat_info[mb,0] + attributes["MeanStatLat"]
                        predicted_val = predicted_val+ stat_info[mb,0] + attributes["MeanStatLat"]
                    if kwargs.get("gt_select")[i] == 'epiLON':
                        true_val =  true_val + stat_info[mb,1] + attributes["MeanStatLon"]
                        predicted_val =  predicted_val + stat_info[mb,1] + attributes["MeanStatLon"]
                    ##################### 
                    true_values[i].append(true_val.item())
                    
                    predictions[i].append(predicted_val)
                    
                    data[i].append((true_val.item(), predicted_val.item()))
                    # Calculate sample-wise errors for test set only
                    if phase == "test": 
                        err_ = true_val.item() - predicted_val.item()
                        column_wise_errors[i].append(err_)
         
        #ave_loss = np.mean(losses) # mean of the batch losses, tüm batchlerin ortalama hata eğeri
        
        # Check if "epiLAT" and "epiLON" are present in gt_select
        if "epiLAT" in gt_select and "epiLON" in gt_select:
            
            # For Spherical Distance Conversion Between Angles and Length, apply Haversine formula
            for idx in range(len(data[0])):
                
                # Compute the haversine distance and append it to the metric_dist list
                distance = haversine(data[gt_select.index("epiLON")][idx][0], 
                                     data[gt_select.index("epiLAT")][idx][0], 
                                     data[gt_select.index("epiLON")][idx][1], 
                                     data[gt_select.index("epiLAT")][idx][1])
                metric_dist.append(distance)
                
        if "Vs30" in gt_select:
            for idx in range(len(data[0])):
                
                diff = data[gt_select.index('Vs30')][idx][0] - data[gt_select.index('Vs30')][idx][1]
                percentage = ((data[gt_select.index('Vs30')][idx][0] - data[gt_select.index('Vs30')][idx][1])/data[gt_select.index('Vs30')][idx][0])*100
                Vs30_diff.append(diff)
                Vs30_percentage.append(percentage)
            wandb.log({"Vs30_diff": np.round(np.mean(np.abs(Vs30_diff)),2)})
            wandb.log({"Vs30_percentage": (np.mean(np.abs(Vs30_percentage)))})
                
        if phase == "training":
            plot_dict = {"groundTruth_training": true_values,
                         "predictions_training": predictions,
                         "training_data": data
                         }
        if phase == "validation":
            plot_dict = {"groundTruth_validation": true_values,
                         "predictions_validation": predictions,
                         "validation_data": data
                         }
        if phase == "test":    
            plot_dict = {"groundTruth_test": true_values,
                         "predictions_test": predictions,
                         "test_data": data,
                         "column_wise_errors":column_wise_errors,
                         "metric_dist":metric_dist,
                         "Vs30_diff":Vs30_diff,
                         "Vs30_percentage":Vs30_percentage
                         }
            
        
        if phase == "test":
            with open(kwargs.get('logs_path'), "a") as file:
                file.write("trainSetSize: " + str(len(attributes["trind"])) + "\n")
                file.write("valSize: " + str(len(attributes["vlind"])) + "\n")
                file.write("testSize: " + str(len(attributes["tsind"])) + "\n")
                file.write("Results:\n")
                file.write("---------------\n")
                file.write("Vs30_Diff Error:")
                file.write(str(np.round(np.mean(np.abs(Vs30_diff)),2)) + "\n")
                file.write("Vs30_Diff Percentage Error:")
                file.write(str(np.mean(np.abs(Vs30_percentage))) + "\n")
                file.write("Metric_dist Error:")
                file.write(str(np.round(np.mean(metric_dist),2)) + "\n")
                file.close()
                
        if 'Vs30' in kwargs['gt_select']: 
            trind = attributes['trind']
            tsind = attributes['tsind']
            vlind = attributes['vlind']
            StatID = attributes['Station ID']
            ZA = []
            ZB = []
            ZC = []
            ZD = []
            ZE = []
            ZA_stat = []
            ZB_stat = []
            ZC_stat = []
            ZD_stat = []
            ZE_stat = []
            ZA_ov = []
            ZB_ov = []
            ZC_ov = []
            ZD_ov = []
            ZE_ov = []
            lat = attributes['stat_info'][:,0]
            lon = attributes['stat_info'][:,1]
            
            if phase == "training":
                with open(kwargs.get('logs_path_Vs30_train'), "a") as file:   
                    
                    for true, pred in zip(true_values, predictions):
                        for i,(a,b, t, p) in zip(trind, zip(lat, lon, pred, true)):
                            a = lat[i]
                            b = lon[i]
                            line = f"{a}\t\t\t{b}\t\t{t}\t\t\t{p}\n"
                            file.write(line)
                    file.close()
                
            if phase == "validation":
                with open(kwargs.get('logs_path_Vs30_val'), "a") as file:   
                    
                    for true, pred in zip(true_values, predictions):
                        for i,(a,b, t, p) in zip(vlind, zip(lat, lon, pred, true)):
                            a = lat[i]
                            b = lon[i]
                            line = f"{a}\t\t\t{b}\t\t{t}\t\t\t{p}\n"
                            file.write(line)
                    file.close()
                        
            if phase == "test":
                with open(kwargs.get('logs_path_Vs30_test'), "a") as file:   
                   
                    for true, pred in zip(true_values, predictions):
                        for i,(a,b, t, p) in zip(tsind, zip(lat, lon, pred, true)):
                            if p > 1500:
                                ZA_stat.append(StatID[i])
                                ZA.append((abs(t-p)/p)*100)
                            elif p > 760 and p < 1500:
                                ZB_stat.append(StatID[i])
                                ZB.append((abs(t-p)/p)*100)
                            elif p > 360 and p < 760:
                                ZC_stat.append(StatID[i])
                                ZC.append((abs(t-p)/p)*100)
                            elif p > 180 and p < 360:
                                ZD_stat.append(StatID[i])
                                ZD.append((abs(t-p)/p)*100)
                            else:
                                ZE_stat.append(StatID[i])
                                ZE.append((abs(t-p)/p)*100)
                            
                            a = lat[i]
                            b = lon[i]
                            line = f"{a}\t\t\t{b}\t\t{t}\t\t\t{p}\n"
                            file.write(line)
                    file.close()
                    
                with open(kwargs.get('logs_path_Vs30_test'), "a") as file:
                    if len(ZA) >0:
                        ZA_ov = sum(ZA)/len(ZA)
                    if len(ZB) >0:
                        ZB_ov = sum(ZB)/len(ZB)
                    if len(ZC) >0:
                        ZC_ov = sum(ZC)/len(ZC)
                    if len(ZD) >0:
                        ZD_ov = sum(ZD)/len(ZD)
                    if len(ZE) >0:
                        ZE_ov = sum(ZE)/len(ZE)
                    
                    line = f"No. of Stations: {len(list(set(ZA_stat)))} \t\t Site A (Absolute Mean Error): {ZA_ov}\n"
                    file.write(line)
                    line2 = f"No. of Stations: {len(list(set(ZB_stat)))} \t\t Site B (Absolute Mean Error): {ZB_ov}\n"
                    file.write(line2)
                    line3 = f"No. of Stations: {len(list(set(ZC_stat)))} \t\t Site C (Absolute Mean Error): {ZC_ov}\n"
                    file.write(line3)
                    line4 = f"No. of Stations: {len(list(set(ZD_stat)))} \t\t Site D (Absolute Mean Error): {ZD_ov}\n"
                    file.write(line4)
                    line5 = f"No. of Stations: {len(list(set(ZE_stat)))} \t\t Site E (Absolute Mean Error): {ZE_ov}\n"
                    file.write(line5)
                    
                    file.close()
                   
    return plot_dict


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r
    