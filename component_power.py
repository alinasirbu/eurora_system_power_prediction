import csv
import gzip
import datetime
import numpy as np
import matplotlib
import pylab as pl
import math
import pickle as pkl
import sys
import math
import time
import string
import matplotlib.dates as dt

month='10'

global_avg=list(csv.reader(open('user_data/avgs_'+month+'_global.csv','r')))
global_avg=np.array(global_avg[0],dtype=np.float)

##load jobs in time
jobs=list(csv.reader(open('system_time_jobs.csv','r')))
jobs=[j for j in jobs if j[0][:4]=='2014' and j[0][5:7]==month]
user_jobs={} ## user_name:(time, job_id)
for j in jobs:
    job_list=j[1].split(';')
    for jj in job_list:
        user_jobs[jj[:8]]=user_jobs.get(jj[:8],[])+[(datetime.datetime.strptime(j[0],'%Y-%m-%d %H:%M:%S UTC'),jj[8:])]


###load power in time
power=list(csv.reader(gzip.open('system_data_nodes_down_mics_idle.csv.gz','r')))
used_power={datetime.datetime.strptime(r[0],'%Y-%m-%d %H:%M:%S UTC'):float(r[7])+float(r[8])+float(r[9])+float(r[10])+float(r[11]) for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==month  }
idle_power={datetime.datetime.strptime(r[0],'%Y-%m-%d %H:%M:%S UTC'):float(r[12])+float(r[13])+float(r[14])+float(r[15])+float(r[16]) for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==month }


predicted_power={r:0 for r in used_power}

###for each user find predicted power usage
for user in user_jobs:
    #look for model results
    try:
        nrmse,rmse,corr,r2,total_power,prediction,jobs,times=pkl.load(gzip.open('job_prediction_results/'+user+'_total.pkl.gz','r'))
        #model exists
        print (user+' found')
        pred_dict={(datetime.datetime.strptime(times[i],'%Y-%m-%d %H:%M:%S UTC'),jobs[i]):prediction[i] for i in range(len(prediction))}           
    except IOError:
        #model does not exist
        print (user+' not found')
        user_avg=list(csv.reader(open('user_data/avgs_'+month+'_'+user+'.csv','r')))
        user_avg=np.array(user_avg[1])
        user_avg[user_avg=='']='0'
        user_avg=np.array(user_avg,dtype=np.float)
        ##where avg is missing use global
        for i in range(len(user_avg)):
            if user_avg[i]==0:
                user_avg[i]=global_avg[i]
        ###compute prediction for this user - store in pred_dict
        pred_dict={}
        job_data=list(csv.reader(gzip.open('user_data/'+user+'_jobs.csv.gz','r')))
        job_data=[r for r in job_data[1:] if r[0][:4]=='2014' and r[0][5:7]==month]
        for job in job_data:
            pred_dict[(datetime.datetime.strptime(job[0],'%Y-%m-%d %H:%M:%S UTC'),job[1])]=sum(user_avg*np.array(job[9:14],dtype=np.float))
    
    
    for j in user_jobs[user]:
        predicted_power[j[0]]+=pred_dict.get(j,0)
           

result=[[i,used_power[i]+idle_power[i],predicted_power[i]+idle_power[i]] for i in used_power if predicted_power[i]>0]
result.sort(key=lambda x:x[0])
result=np.array(result)
result=result[result[:,0]<datetime.datetime(2014,int(month),15),:] #remove last part for plotting reasons#last 8 are from 30th, with huge gap, we remove them

MAE=math.sqrt(np.average((result[:,1]-result[:,2])**2))/np.mean(np.array(result[:,1],dtype=np.float))
r2=1-(sum((result[:,2]-result[:,1])**2)/sum((result[:,1]-np.mean(np.array(result[:,1],dtype=np.float)))**2))
pl.figure(figsize=(10,4))

pl.plot(result[:,0],result[:,2],color='orange',linewidth=2)
pl.plot(result[:,0],result[:,1],color='black',ls=':')
pl.legend(['Predicted','Real'],loc=3,frameon=False)
pl.ylabel('Computing Power (W)')
pl.xlabel('Date (2014)')
pl.ylim(pl.ylim()[0]*0.95,pl.ylim()[1])
locs, labels=pl.xticks()
labels=[dt.num2date(l).strftime('%b %d') for l in locs]
pl.xticks(locs,labels)
pl.xlim(pl.xlim()[0],735513.75)
pl.subplots_adjust(bottom=0.2)
pl.title('NRMSE='+str(int(MAE*1000)/1000.0)+' R-squared='+str(int(r2*1000)/1000.0))
pl.savefig('component_power'+month+'.pdf', format='pdf')

pkl.dump(result,file=gzip.open('predicted_component_power'+month+'.pkl.gz','w'))
