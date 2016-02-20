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
from sklearn import linear_model
import time
import string

test_month='10'
train_month='09'


###load power in time

power=list(csv.reader(gzip.open('system_data_nodes_down_mics_idle.csv.gz','r')))

train_component_power=[[sum(map(float,r[7:17]))] for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==train_month ]
test_component_power=[[sum(map(float,r[7:17]))] for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==test_month ]

train_system_power=[float(r[1]) for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==train_month ]
test_system_power=[float(r[1]) for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==test_month ]


train_time=[datetime.datetime.strptime(r[0],'%Y-%m-%d %H:%M:%S UTC') for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==train_month]
test_time=[datetime.datetime.strptime(r[0],'%Y-%m-%d %H:%M:%S UTC') for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==test_month]


regr = linear_model.LinearRegression()
regr.fit(train_component_power, train_system_power)
predicted_system_power=regr.predict(test_component_power)

test_result=[[test_time[i],test_system_power[i],predicted_system_power[i]] for i in range(len(test_time))]
test_result.sort(key=lambda x:x[0])
test_result=np.array(test_result)
test_result=test_result[test_result[:,0]<datetime.datetime(2014,int(test_month),15),:] #remove last part for plotting reasons

MAE=math.sqrt(np.average((test_result[:,1]-test_result[:,2])**2))/np.mean(np.array(test_result[:,1],dtype=np.float))
r2=1-(sum((test_result[:,2]-test_result[:,1])**2)/sum((test_result[:,1]-np.mean(np.array(test_result[:,1],dtype=np.float)))**2))
pl.figure(figsize=(10,4))

pl.plot(test_result[:,0],test_result[:,2],color='orange',linewidth=2)
pl.plot(test_result[:,0],test_result[:,1],color='black',ls=':')
pl.legend(['Modeled system power','Real system power'],loc='best')
pl.ylabel('Power (W)')
pl.xlabel('Date')
pl.xticks(rotation=30)
pl.subplots_adjust(bottom=0.3)
pl.title('NRMSE='+str(int(MAE*1000)/1000.0)+' R-squared='+str(int(r2*1000)/1000.0))
pl.savefig('system_power'+test_month+'.pdf', format='pdf')

pkl.dump(regr,file=gzip.open('model'+test_month+'.pkl.gz','w'))
pkl.dump(test_result,file=gzip.open('linear_model_test'+test_month+'.pkl.gz','w'))
