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
import matplotlib.dates as dt
from sklearn import linear_model
import time
import string

test_month='10'



###load power in time

power=list(csv.reader(gzip.open('system_data_nodes_down_mics_idle.csv.gz','r')))

system_power={datetime.datetime.strptime(r[0],'%Y-%m-%d %H:%M:%S UTC'):float(r[1]) for r in power[1:] if r[0][:4]=='2014' and r[0][5:7]==test_month }

predictions=pkl.load(gzip.open('predicted_component_power'+test_month+'.pkl.gz','r'))

predicted_times=predictions[:,0]
predicted_component_power=predictions[:,2]

model=pkl.load(gzip.open('model'+test_month+'.pkl.gz','r'))

predicted_system_power=model.predict([[cp] for cp in predicted_component_power])

test_result=[[predicted_times[i],system_power[predicted_times[i]],predicted_system_power[i]] for i in range(len(predicted_times))]
test_result.sort(key=lambda x:x[0])
test_result=np.array(test_result)
test_result=test_result[test_result[:,0]<datetime.datetime(2014,int(test_month),15),:] #remove last part for plotting reasons

MAE=math.sqrt(np.average((test_result[:,1]-test_result[:,2])**2))/np.mean(np.array(test_result[:,1],dtype=np.float))
r2=1-(sum((test_result[:,2]-test_result[:,1])**2)/sum((test_result[:,1]-np.mean(np.array(test_result[:,1],dtype=np.float)))**2))
pl.figure(figsize=(10,4))

pl.plot(test_result[:,0],test_result[:,2],color='orange',linewidth=2)
pl.plot(test_result[:,0],test_result[:,1],color='black',ls=':')
pl.legend(['Predicted','Real'],loc=3,frameon=False)
pl.ylabel('System Power (W)')
pl.xlabel('Date (2014)')
locs, labels=pl.xticks()
labels=[dt.num2date(l).strftime('%b %d') for l in locs]
pl.xticks(locs,labels)

pl.xlim(pl.xlim()[0],735513.75)
pl.subplots_adjust(bottom=0.2)
pl.title('NRMSE='+str(int(MAE*1000)/1000.0)+' R-squared='+str(int(r2*1000)/1000.0))
pl.savefig('system_power'+test_month+'_pred.pdf', format='pdf')


pkl.dump(test_result,file=gzip.open('applied_linear_model_test'+test_month+'.pkl.gz','w'))
