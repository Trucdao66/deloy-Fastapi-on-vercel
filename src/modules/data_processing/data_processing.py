from fastapi.exceptions import HTTPException
import io
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from ..check_status.models.models import gauss, two_gauss, bac4, tong
import matplotlib.pyplot as plt

def weighted_moving_average(column, weight):
    if weight <= 0:
        raise HTTPException(status_code=400, detail="weight must be a positive integer")
    weights = np.arange(1, weight)
    wma = np.convolve(column, weights, mode='valid') / weights.sum()
    return wma

def show_data(contents_bytes: bytes,weight:int, background: bool) -> dict:
    if not background in [True, False]:
        raise HTTPException(status_code=400, detail="weight must be a positive integer")
    contents_buffer = io.BytesIO(contents_bytes)
    df = pd.read_csv(contents_buffer, sep='\t', header=None)
    details = {}
    if df.shape[1] == 1:
        resuls = weighted_moving_average(df.iloc[:, 0].values,weight)
        details = {'only a conlunm' : resuls.tolist()}
    elif background:
        df = df.iloc[:, 1:].subtract(df.iloc[:, 0], axis=0) #lấy cột đầu tiên là phông
        for column_name in df.columns:
            resuls = weighted_moving_average(df[column_name],weight)
            column_values = resuls.tolist()
            details[column_name] = column_values
    else:
        for column_name in df.columns:
            resuls = weighted_moving_average(df[column_name],weight)
            column_values = resuls.tolist()
            details[column_name] = column_values
    return details

def FIT (deatails:dict,cantren:int,canduoi:int):
    if cantren <= canduoi:
        raise HTTPException(status_code=400, detail='The value of cantren must be greater than the value of canduoi')
    column = {}
    for column_name in deatails:
        column_values = deatails[column_name][canduoi:cantren]
        xdata = np.array(range(len(column_values)))
        mean = np.mean(column_values)
        std = np.std(column_values)
        parameters, covariance = curve_fit(gauss, xdata, column_values,p0=[1/(std*np.sqrt(2*np.pi)),mean,std],maxfev = 1000000)
        deatail = [mean,std]
        fity=gauss(xdata,*parameters)
        fity_aray= np.array(fity)
        column["data"] = column_values
        column["mean and sigma"] = deatail
        column["data after fit"] = fity_aray.tolist()
        deatails[column_name] = column
    return {f"area_fit(low:{canduoi},high:{cantren}),function_fit is gauss. Data, data after fit, mean and sigma of each column in the file ": deatails}

def scattering (deatails:dict):
    column = {}
    for column_name in deatails:
        ydata =  deatails[column_name]
        n = len(ydata)
        xdata = np.array(range(n))
        parameters, covariance = curve_fit( tong, xdata, ydata ,p0=[0,0,0,0,0,50,750,10,50,1250,10,800,1100,10],maxfev = 5000)
        perr = np.sqrt(np.diag(covariance))
        #plt.plot(xdata,ydata,label = "data")
        #plt.plot(xdata,two_gauss(xdata,*parameters[5:11]),label ='TX 2L')
        #plt.plot(xdata,gauss(xdata,*parameters[5:8]),label = "tan xa 2.1")
        #plt.plot(xdata,gauss(xdata,*parameters[8:11]),label = "tan xa 2.2")
        #plt.plot(xdata,gauss(xdata,*parameters[11:]),label = "tan xa 1")
        #plt.plot(xdata[500:],bac4(xdata[500:],*parameters[:5]))

        z = ydata - two_gauss(xdata,*parameters[5:11]) - gauss(xdata,*parameters[11:])
        cof=np.polyfit(xdata,z,4)
        b4 = np.poly1d(cof)
        #print(b4)
        #plt.plot(xdata,b4(xdata),label = "tan xa tren 2L")
    
        #plt.plot(xdata,two_gauss(xdata,*parameters[5:11]) + gauss(xdata,*parameters[11:]) + b4(xdata),label = "fit tong")
        mean_tx1,sigma_tx1 = parameters[12],parameters[13]
        #print(f"mean và do lech chuan cua tan xa 1:{mean_tx1,sigma_tx1}")
        #print(f"dien tich dinh pho tan xa 1:{sum(gauss(xdata,*parameters[11:])[int(mean_tx1-3*sigma_tx1):int(mean_tx1+3*sigma_tx1)])}")
        #print(mean_tx1+3*sigma_tx1,mean_tx1-3*sigma_tx1)
        deatails[column_name] = column
        column["mean tx 1"] = mean_tx1
        column["do lech chuan cua tx 1"] = sigma_tx1
        column["Dien tich dinh pho tx1"] = sum(gauss(xdata,*parameters[11:])[int(mean_tx1-3*sigma_tx1):int(mean_tx1+3*sigma_tx1)])
    #plt.legend()
    #plt.show()
    return {"infor processing scattering":deatails}



