import prediction as pred
import tkinter as tk
from tkinter import filedialog
from itertools import islice
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colorbar as mplcb
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import cProfile, pstats, io
import copy
import csv
import mmap
import os

__author__ = 'Pierce Rixon'
select = 1 #up to 5

null = None


def main():
    #ws='whitespace'
    #ts='timescale'
    #f='frequency'
    #bw='bandwidth'
    #fn='frame_no'
    

    print('Running Analysis Suite')

    matplotlib.style.use('ggplot')

    root = tk.Tk()
    root.withdraw()
    
    #Use metadata to strip the 'Total whitespace' message out of the end of each csv file selected
    #metadata()

    print('Select configuration file')
    cfgfile = filedialog.askopenfilename(filetypes = [("config files","*.cfg")])

    with open(cfgfile) as cfg:
        config = cfg.read().splitlines()

    #occupancy(config)
    #occupancy_v2(config)
    occupancy_csv(config) #NOT WORKING FOR LARGE DATASETS 100MB+
    #dutycycle(config)
    #PSD(config)

    #print('Select window file')
    filename = filedialog.askopenfilename(filetypes = [("csv files","*.csv")]) #('Window_dump.csv')    
    
    #pred.pred(filename)

    #hexbin(filename)
    #analyse(filename,select) #For analysis package
    #dev_sim(filename) #For device simulator
    #legacy_sns(filename)
    #legacy(filename)



def metadata():
    
    print('Select files to remove \'total whitespace\' message and dump to metadata file')
    print('because im lazy, if the file does not have \'total whitespace\' at the end, it will take 5ever for the function to return')
    filenames = filedialog.askopenfilenames(filetypes = [("csv files","*.csv")])
    filelist = list(filenames)

    mpath = os.path.dirname(filelist[0])+'/metadata.txt'

    print(mpath)

    print('number of files: {}'.format(len(filelist)))
    #print(filelist)

    meta_file = open(mpath, 'w+')
    meta_file.write('Number of files: {}\n'.format(len(filelist)))

    for f in filelist:
        meta_file.write(f)
  #      meta_file.write('\n')
        with open(f, 'r+') as file:
            file.seek(0, os.SEEK_END)
            pos = file.tell() - 1
            end = file.tell() - 1

            while pos > 0 and file.read(1) != "T":
                pos -= 1
                file.seek(pos, os.SEEK_SET)

            pos -= 1
            file.seek(pos, os.SEEK_SET)

            if pos > 0:
                string = file.read(end-pos)
                print(string)
                meta_file.write(string+'\n\n')
                file.seek(pos, os.SEEK_SET)
                file.truncate()
            else:
                print('File does not have Total Whitespace message')
            file.close()

def PSD(config):
    print('Now running PSD plot, ensure a PSD dataset is selected')
    filename = filedialog.askopenfilename(filetypes = [("csv files","*.csv")])
    dataset = pd.read_csv(filename, header=0)

    c_freq = np.float(config[2])
    #config[3] has the filter bandwidth, which is not active for a WBX daughterboard, as its locked to 40MHz
    c_bandwidth = np.float(config[4])

    print(dataset)

    d_len = dataset['avg'].size

    print(d_len)

    freq = np.linspace(0,d_len-1,d_len)
    
    fig = plt.figure(figsize=(22,10))
    fig.subplots_adjust(left=0.08, bottom=0.12, right=.97, top=.95)
    ax = fig.add_subplot(1,1,1) 

    for spine in ['left','right','top','bottom']:
       ax.spines[spine].set_color('k')
 
    ax.tick_params(which = 'major', width=1, length=3, color='k')
    ax.tick_params(which = 'minor', width=.5, length=1, color='k')
         
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    start = (c_freq - (c_bandwidth*.5))/1e6
    stop = (c_freq + (c_bandwidth*.5))/1e6
    ax.set_xticklabels(np.linspace(start,stop,11))
    ax.set_xticks(np.linspace(0,d_len-1,11))
    ax.set_xlim(0,d_len-1)
    ax.set_ylim(-130,-50)

    #ax.set_title('Power Spectrum Density', fontsize=36)
    ax.set_xlabel('Frequency (MHz)', fontsize=(36+6))
    ax.set_ylabel('Power (dBm)', fontsize=(36+6))
    ax.tick_params(axis='both', labelsize=(26+6))

    ax.plot(freq,dataset['max'], color='r', label="Max")
    ax.plot(freq,dataset['avg'], color='k', label="Avg")
    ax.plot(freq,dataset['min'], color='b', label="Min")

    ax.axvline(x=d_len*.1, color='k', linestyle='--')
    ax.axvline(x=d_len*.9, color='k', linestyle='--')

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles,labels,fontsize=24)
    plt.rc('axes', labelsize=(36+6))   
    plt.show()  

    # here we are plotting stacked slices on top of eachother, plotting frequency vs time (in frames)
def occupancy_csv(config):
     print('Now running occupancy plot, ensure a bandwidth dataset is selected')
     filename = filedialog.askopenfilename(filetypes = [("csv files","*.csv")])
     
     with open(filename, 'r') as r:
         next(r)
         dataset = [[int(x) for x in rec] for rec in csv.reader(r, delimiter=',')]
     #dataset = pd.read_csv(filename, header=0)

     c_freq = np.float(config[2])
     #config[3] has the filter bandwidth, which is not active for a WBX daughterboard, as its locked to 40MHz
     c_bandwidth = np.float(config[4])

     resolution = 131072 #this will have to be modified depending on how the FFT is computed

     framemax = np.amax(dataset, axis=0)[4] #frame_no
     print(framemax)
     #framemax = np.max(dataset['frame_no'])

     hslices = 512 #number of horizontal slices
     vslices = 1024 #number of vertical slices
     
     hscans = int(framemax/hslices) + 1
     vscans = int((resolution*0.8)/vslices) #this should be an integer, if it isnt, tough :)

     freq_arry = np.zeros(resolution*0.8)
     indent = int(resolution * 0.1)

     mesh = np.zeros((hslices+1,vslices))

     hidx = 0
     count = 0

     #row:idx,timescale,frequency,bandwidth,whitespace,frame_no
     for row in dataset:
        #as the rows are sorted by frame number, we can just iterate through them
        if row[4] > (hidx+1)*hscans:

            #zero array
            if hidx%100 == 0 : print(hidx)
            for i in range(vslices):
                mesh[hidx,i] = (np.sum(freq_arry[i*vscans:((i+1)*vscans - 1)])/hscans - (vscans-1))*-1
            hidx = hidx + 1
            freq_arry.fill(0)
        
        count = count + 1
        freq_arry[row[1] - indent : row[1]+row[2]-1 - indent] += 1

        if count%100000 == 0:
            print(count)

    #drop lastrow
     #for i in range(vslices):
     #   mesh[hidx,i] = (np.sum(freq_arry[i*vscans:((i+1)*vscans - 1)])/hscans - vscans)*-1
     fig = plt.figure()
     ax = fig.add_subplot(1,1,1) 
     
     for spine in ['left','right','top','bottom']:
        ax.spines[spine].set_color('k')
 
     ax.tick_params(which = 'major', width=1, length=3, color='k')
     ax.tick_params(which = 'minor', width=.5, length=1, color='k')
         
     ax.yaxis.set_ticks_position('left')
     ax.xaxis.set_ticks_position('bottom')

        #plt.subplot(111)
     meshm = np.ma.masked_where(mesh < 1, mesh)
     m = ax.pcolormesh(meshm, vmin=1, vmax=np.amax(mesh), cmap='gnuplot_r')
     
     ax.axis([0,vslices,0,hidx])
     ax.set_xticks(np.linspace(1,vslices,5))#, 26316, 52632])

     start = (c_freq - (c_bandwidth*.4))/1e6
     stop = (c_freq + (c_bandwidth*.4))/1e6
     ax.set_xticklabels(np.linspace(start,stop,5)/60)
     #ax.set_ticks(True)
     
     
     plt.colorbar(m,ax=ax)
     #plt.grid(True, which='major', axis='both', linestyle='-', color='none')
     
     ax.set_title('Spectrum Occupancy')
     ax.set_xlabel('Frequency (MHz)')
     ax.set_ylabel('Time (min)')

     ax.set_yticks(np.linspace(0,hidx,num=12))
     ax.set_yticklabels(np.around(np.linspace(0,hidx,num=12)*.005*hscans,decimals=2))

     plt.tight_layout()
     plt.show()

def occupancy(config):
     print('Now running occupancy plot, ensure a bandwidth dataset is selected')
     filename = filedialog.askopenfilename(filetypes = [("csv files","*.csv")])
     dataset = pd.read_csv(filename, header=0)

     c_freq = np.float(config[2])
     #config[3] has the filter bandwidth, which is not active for a WBX daughterboard, as its locked to 40MHz
     c_bandwidth = np.float(config[4])

     resolution = 131072 #this will have to be modified depending on how the FFT is computed

     framemax = np.max(dataset['frame_no'])

     hslices = 512 #number of horizontal slices
     vslices = 1024 #number of vertical slices
     
     hscans = int(framemax/hslices) + 1
     vscans = int((resolution*0.8)/vslices) #this should be an integer, if it isnt, tough :)

     freq_arry = np.zeros(resolution*0.8)
     indent = int(resolution * 0.1)

     mesh = np.zeros((hslices+1,vslices))

     hidx = 0
     count = 0

     #row:idx,timescale,frequency,bandwidth,whitespace,frame_no
     for row in dataset.itertuples():
        #as the rows are sorted by frame number, we can just iterate through them
        if row[5] > (hidx+1)*hscans:

            #zero array
            if hidx%100 == 0: print(hidx)
            for i in range(vslices):
                mesh[hidx,i] = (np.sum(freq_arry[i*vscans:((i+1)*vscans - 1)])/hscans - (vscans-1))*-1
            hidx = hidx + 1
            freq_arry.fill(0)
        
        count = count + 1
        freq_arry[row[2] - indent : row[2]+row[3]-1 - indent] += 1

        if count%100000 == 0:
            print(count)

    #drop lastrow
     #for i in range(vslices):
     #   mesh[hidx,i] = (np.sum(freq_arry[i*vscans:((i+1)*vscans - 1)])/hscans - vscans)*-1
     fig = plt.figure(figsize=(22,10))
     fig.subplots_adjust(left=0.07, bottom=0.12, right=1.08, top=.97)
     ax = fig.add_subplot(1,1,1)     
     
     for spine in ['left','right','top','bottom']:
        ax.spines[spine].set_color('k')
 
     ax.tick_params(which = 'major', width=1, length=3, color='k')
     ax.tick_params(which = 'minor', width=.5, length=1, color='k')
         
     ax.yaxis.set_ticks_position('left')
     ax.xaxis.set_ticks_position('bottom')

        #plt.subplot(111)
     meshm = np.ma.masked_where(mesh < 1, mesh)
     m = ax.pcolormesh(meshm, vmin=1, vmax=np.amax(mesh), cmap='inferno_r')
     
     ax.axis([0,vslices,0,hidx])
     ax.set_xticks(np.linspace(1,vslices,5))#, 26316, 52632])

     start = (c_freq - (c_bandwidth*.4))/1e6
     stop = (c_freq + (c_bandwidth*.4))/1e6
     ax.set_xticklabels(np.linspace(start,stop,5))
     #ax.set_ticks(True)
     
     cb = plt.colorbar(m,ax=ax)
     #labelstr = 'Observation Density (p/'+str(hscans*vscans)+')'
     labelstr = 'Observation Density'
     cb.set_label(labelstr, fontsize=(26+6))
     cb.ax.tick_params(labelsize=(20+6))
     #plt.grid(True, which='major', axis='both', linestyle='-', color='none')
     
     #ax.set_title('Spectrum Occupancy',fontsize=)
     ax.set_xlabel('Frequency (MHz)', fontsize=(36+6))
     ax.set_ylabel('Time (min)', fontsize=(36+6))
     ax.tick_params(axis='both', labelsize=(26+6))

     ax.set_yticks(np.linspace(0,hidx,num=11))
     #ax.set_yticklabels(np.around(np.linspace(0,hidx,num=12)*.005*hscans/60,decimals=1))
     ax.set_yticklabels(np.around(np.linspace(0,5,num=11),decimals=1))

     plt.rc('axes', labelsize=(32+6)) 
     plt.show()

def occupancy_v2(config):

#Occupancy is a measure of duration
     print('Now running occupancy plot. This will work for any partitioned dataset')
     filename = filedialog.askopenfilename(filetypes = [("csv files","*.csv")])
     dataset = pd.read_csv(filename, header=0)

     c_freq = np.float(config[2])
     #config[3] has the filter bandwidth, which is not active for a WBX daughterboard, as its locked to 40MHz
     c_bandwidth = np.float(config[4])

     resolution = 131072 #this will have to be modified depending on how the FFT is computed

     framemax = np.max(dataset['frame_no'])

     hslices = 1001 #number of horizontal slices
     vslices = 2001 #number of vertical slices
     
     hscans = int(framemax/hslices) + 1
     vscans = np.ceil((resolution*0.8)/vslices) #this should be an integer, if it isnt, tough :)

     indent = int(resolution * 0.1)

     #mesh[row,cols]
     mesh = np.full((hslices-1,vslices-1),np.int(hscans*vscans), dtype=np.int)

     count = 0

     #Condition the dataframe accordingly
     dataset['start'] = dataset['frame_no'].subtract(dataset['timescale']-1)
     dataset['frequency'] = dataset['frequency'].subtract(indent)
     
     print(dataset)
     print("hscans: {}".format(hscans))
     print("vscans: {}".format(vscans))
     #      0      1        2           3       4         5       6
     #row:idx,timescale,frequency,bandwidth,whitespace,frame_no,start
     for row in dataset.itertuples():


        #try slicing lel

        #mesh[np.int((row[6] - 1)/hscans) : np.int((row[2] - 1)/vscans), \
        #    np.int((row[6] + row[1] - 1)/hscans) : np.int((row[2] + row[3] - 1)/vscans)] += 1

        x = np.floor(row[2]/vscans)
        y = np.floor(row[6]/hscans)

        d = np.int(np.ceil(row[1]/hscans))
        b = np.int(np.ceil(row[3]/vscans))

        for i in range(d):
            for j in range(b):

                cell = vscans * hscans

                #weight = 
                w_x = 0
                w_y = 0

                w_xl = row[2] - (x + j) * vscans                
                w_xh = row[2] + row[3]-1 - (x + j + 1) * vscans

                if w_xl > 0 and w_xh > 0:
                    w_x = vscans - w_xl + 1
                elif w_xl > 0 and w_xh <= 0:
                    w_x = row[3]
                elif w_xh > 0 and w_xl <= 0:
                    w_x = vscans
                else: 
                    w_x = row[2] + row[3]-1 - (x + j) * vscans

                w_yl = row[6] - (y + i) * hscans
                w_yh = row[6] + row[1]-1 - (y + i + 1) * hscans

                if w_yl > 0 and w_yh > 0:
                    w_y = hscans - w_yl + 1
                elif w_yl > 0 and w_yh <= 0:
                    w_y = row[1]
                elif w_yh > 0 and w_yl <= 0:
                    w_y = hscans
                else:
                    w_y = row[6] + row[1]-1 - (y + i) * hscans
                
                #print(row)
                #print("x:{},y:{}  j{},i{}  wx{} wy{}".format(x,y,j,i,w_x,w_y))

                mesh[y + i, x + j] -= w_y*w_x
        
        count = count + 1
       
        if count%100 == 0:
            print(count)



     #Determine the starting time of a partition

     #print("Start added to dataset \n {}".format(dataset))

     ##perform thresholding
     #dataset['start'] = np.ceil(dataset['start'].div(hscans))
     ##dataset['start'] = np.ceil(dataset['start'])    
 
     #dataset['frequency'] = np.ceil(dataset['frequency'].subtract(indent).div(vscans))
     ##dataset['frequency'] = np.ceil(dataset['frequency'])
 
     #dataset['bandwidth'] = np.ceil(dataset['bandwidth'].div(vscans))
     ##dataset['bandwidth'] = np.ceil(dataset['bandwidth'])
 
     #dataset['timescale'] = np.ceil(dataset['timescale'].div(hscans))
     ##dataset['timescale'] = np.ceil(dataset['timescale'])


     #dataset = dataset.sort_values(['frequency','start'], ascending=True)

     #print("Thresholded dataset \n {}".format(dataset))

     #dataset_g = dataset.groupby(['start','frequency','timescale','bandwidth']).size().reset_index().rename(columns={0:'count'})

     #print(dataset_g)

     ##dataset = dataset.sort_values(['frequency','start'], ascending=True).reset_index()
     
     ##print("Sorted by frequency \n {}".format(dataset))
     ##Grouping must be the last operation, otherwise numpy functions will not work correctly



     ##row:idx,start,frequency,timescale,bandwidth,count
     #for row in dataset_g.itertuples():
        
     #   for i in range(np.int(row[3])):
     #       for j in range(np.int(row[4])):
     #           mesh[np.int(row[1]) + i - 1, np.int(row[2]) + j - 1] += np.int(row[5])
     #   count += 1

     #   if count%10000 == 0:
     #       print(count)

    #drop lastrow
     #for i in range(vslices):
     #   mesh[hidx,i] = (np.sum(freq_arry[i*vscans:((i+1)*vscans - 1)])/hscans - vscans)*-1
     fig = plt.figure()
     ax = fig.add_subplot(1,1,1) 
     
     for spine in ['left','right','top','bottom']:
        ax.spines[spine].set_color('k')
 
     ax.tick_params(which = 'major', width=1, length=3, color='k')
     ax.tick_params(which = 'minor', width=.5, length=1, color='k')
         
     ax.yaxis.set_ticks_position('left')
     ax.xaxis.set_ticks_position('bottom')

        #plt.subplot(111)
     meshm = np.ma.masked_where(mesh < 1, mesh)
     m = ax.pcolormesh(meshm, vmin=1, vmax=np.amax(mesh), cmap='gnuplot_r')
     
     ax.axis([0,vslices,0,hslices])
     ax.set_xticks(np.linspace(1,vslices,5))#, 26316, 52632])

     start = (c_freq - (c_bandwidth*.4))/1e6
     stop = (c_freq + (c_bandwidth*.4))/1e6
     ax.set_xticklabels(np.linspace(start,stop,5))
     #ax.set_ticks(True)
     
     
     plt.colorbar(m,ax=ax)
     #plt.grid(True, which='major', axis='both', linestyle='-', color='none')
     
     ax.set_title('Spectrum Occupancy')
     ax.set_xlabel('Frequency (MHz)')
     ax.set_ylabel('Time (s)')

     #ax.set_yticks(np.linspace(0,hidx,num=12))
     #ax.set_yticklabels(np.around(np.linspace(0,hidx,num=12)*.005*hscans,decimals=2))

     plt.tight_layout()
     plt.show()



#new improved spicy hexbin with distributions :D
def hexbin(filename):
     print('Now running hexbin plot, ensure a partitioned dataset is selected')
     dataset = pd.read_csv(filename, header=0)

     #matplotlib.rcParams.update({'font.size': 21}
     #plt.figure(figsize=(6,8), dpi=600)

     #xmax = np.power(10,np.ceil(np.log10(np.max(dataset['bandwidth']))))*.5
     bwmax = np.max(dataset['bandwidth'])
     print(bwmax*190)
     if np.power(10,np.ceil(np.log10(bwmax))) > 4*bwmax:
        xmax = np.power(10,np.ceil(np.log10(bwmax))) / 4
     elif np.power(10,np.ceil(np.log10(bwmax))) > 2*bwmax:
        xmax = np.power(10,np.ceil(np.log10(bwmax))) / 2
     else: 
        xmax = np.power(10,np.ceil(np.log10(bwmax)))

     if xmax < 25000: xmax = 25000 #This is here as 4.75MHz is the minimax (smallest max) x axis value. 
     
     print(xmax*190)

#     xmin = 1  #Just for the BW=1 TS=1 dataset
#     ymin = 1
     xmin = 50
     #ymax = np.maximum(1e6,np.power(10,np.ceil(np.log10(np.max(dataset['timescale'])))))
     ymax = np.power(10,np.ceil(np.log10(np.max(dataset['timescale'])))) #for 5min segment
     ymin = np.min(dataset['timescale'])

     ##cheekybonusbits
     #array = np.histogram(dataset['timescale'], bins=np.logspace(np.log10(ymin), np.log10(ymax), num=200))
     #print(array)
     #x = array[1]
     #y = array[0]
     #print(x)
     #print(y)
     #cr_y = np.cumsum(y[::-1])[::-1] 
     #print(cr_y)
     #print("len x: {}, len y: {}".format(len(x[:-1]),len(cr_y)))
     #plt.figure()
     #ax = plt.subplot(111)
     #ax.scatter(x[:-1],cr_y)
     ##ax.set_yscale('log')
     #ax.set_xscale('log')
     #plt.show()

     #create figure hook, assign figure size and adjust padding around borders
     fig = plt.figure(figsize=(22,14))
     fig.subplots_adjust(left=0.1, bottom=0.09, right=.98, top=.98)

     #arrange the various axes nicely using gridspec
     gs=gridspec.GridSpec(5,6)
     gs.update(wspace = 0.03, hspace= 0.03)

     ax1 = plt.subplot(gs[1:,:-1])
     ax2 = plt.subplot(gs[1:,-1])
     ax3 = plt.subplot(gs[0,:-1])
     ax4 = gs[0,-1]

     gs2=gridspec.GridSpecFromSubplotSpec(12,16,ax4)
     cbax = plt.subplot(gs2[4,1:-1])

     #create the hexbin plot and a hook for the colourmap
     cbmap = ax1.hexbin(dataset['bandwidth'], dataset['timescale'], mincnt=1, xscale='log', yscale='log', cmap='inferno', norm=matplotlib.colors.LogNorm(), reduce_C_function=np.sum)
     ax1.axis([xmin, xmax, ymin, ymax])
     
     #populate the appropriate spines
     for spine in ['left','bottom']:
        ax1.spines[spine].set_color('k')
        
     ax2.spines['bottom'].set_color('k')
     ax3.spines['left'].set_color('k')
     
     #set tick sizes, colours and lengths
     ax1.tick_params(which = 'major', width=1, length=4, color='k')
     ax1.tick_params(which = 'minor', width=1, length=2, color='k')

     #manual xtick placement
     ax1.set_xticks([66, 132, 263, 526, 1316, 2632, 5263, 13158])#, 26316])#, 52632])
     ax1.set_xticklabels([r'12.5kHz', r'25kHz', r'50kHz', r'100kHz', r'250kHz', r'500kHz', r'1MHz', r'2.5MHz', r'5MHz', r'10MHz'])
     ax1.xaxis.set_ticks_position('bottom')
     ax1.set_xlabel('Bandwidth', fontsize=(24+18))
     
     #manual ytick placement
     ax1.set_yticks([10,19,48,95,190,477,1907,5722,11444,57220])#,171661,686646])
     ax1.set_yticklabels([r'50ms',r'100ms',r'250ms',r'500ms',r'1s',r'2.5s',r'10s',r'30s',r'1m',r'5m',r'15m',r'1h'])
     ax1.yaxis.set_ticks_position('left')
     ax1.set_ylabel('Duration', fontsize=(24+18))
     ax1.tick_params(axis='both', labelsize=(16+18))
     
     #ax1.set_title("With a log color scale")
     #cbax = mplcb.make_axes_gridspec(ax4)

     cb = plt.colorbar(cbmap,cax=cbax, orientation='horizontal')
     cb.outline.set_visible(True)
     cb.outline.set_edgecolor('black')   
     cb.set_label('Observation Density', fontsize=(14+12))
     cbax.xaxis.set_label_position('top')
     cb.ax.tick_params(labelsize=(14+12))
     #cb.set_ticks([np.log10(1),np.log10(10),np.log10(50),np.log10(100),np.log10(500)])
     #cb.set_ticklabels([1,10,50,100,500])

     #Timescale histogram     
     ax2hist = ax2.hist(dataset['timescale'], bins=np.logspace(np.log10(ymin), np.log10(ymax), num=100), orientation='horizontal', log=True, color='k')
     ax2.set_yscale('log')
     ax2.set_xlim(1,np.power(10,np.ceil(np.log10(np.amax(ax2hist[0])))))

     ax2.yaxis.set_visible(False)
     ax2.xaxis.set_ticks_position('bottom')
     ax2.tick_params(which = 'major', width=1, length=4, color='k')
     ax2.tick_params(which = 'minor', width=1, length=2, color='k')
     
     ax2.tick_params(axis='both', labelsize=(14+18))

     #Better bandwidth histogram
     ax3.hist(dataset['bandwidth'], bins=np.logspace(np.log10(xmin), np.log10(xmax), num=200), log=True, color='k')
     ax3.set_xscale('log')
     ax3.set_xlim(xmin, xmax)
     #ax3.axis([xmin, xmax, 1, ymax])
     
     ax3.xaxis.set_visible(False)
     ax3.yaxis.set_ticks_position('left')
     ax3.tick_params(which = 'major', width=1, length=4, color='k')
     ax3.tick_params(which = 'minor', width=1, length=2, color='k')

     ax3.tick_params(axis='both', labelsize=(14+18))

     plt.rc('axes', labelsize=(20+18))   
     #dpi=plt.gcf().dpi is CRITICAL for saving an image that looks identical to the one displayed in plt.show()
     #plt.savefig('Basic.png', dpi=plt.gcf().dpi)
     plt.show()

     # here we are plotting totals of duration per frequency as a percentage of total duration
def dutycycle(filename):
     print('Now running dutycycle plot, ensure a duration dataset is selected')
     filename = filedialog.askopenfilename(filetypes = [("csv files","*.csv")])
     dataset = pd.read_csv(filename, header=0, dtype=np.int)
 
     resolution = 131072 #this will have to be modified depending on how the FFT is computed

     framemax = np.max(dataset['frame_no'])

     vslices = 1024
     vscans = int((resolution*0.8)/vslices)+1 #this should be an integer, if it isnt, tough :)
     
     newrez = int(resolution*0.8) + 1
     
     freq_arry = np.zeros(newrez)
     indent = int(resolution * 0.1)

     count = 0
     #row:idx,timescale,frequency,bandwidth,whitespace,frame_no
     for row in dataset.itertuples():
         #freq_arry[int((row[2] - indent)/vscans)] += row[1]
         freq_arry[row[2] - indent] += row[1]
         count = count + 1
         
         if count%100000 == 0:
            print(count)
    
     #now manipulate freq_arry, yes its a roundabout way, but ohwell
     #normalise
     freq_arry /= framemax #normalise
     freq_arry *= 100 #out of 100
     freq_arry -= 100 #whitespace is positive, so incumbent activity here will now be negative
     freq_arry *= -1 #make incumbent activity positive

     freq_max = np.zeros(vslices)
     freq_avg = np.zeros(vslices)
     for i in range(newrez):
         if freq_arry[i] > freq_max[int(i/vscans)]:
             freq_max[int(i/vscans)] = freq_arry[i]

         freq_avg[int(i/vscans)] += freq_arry[i]/vscans

 
     fig = plt.figure()
     ax = fig.add_subplot(1,1,1) 

     freq = np.linspace(0,vslices-1,vslices)
#     ax.scatter(freq,freq_max, edgecolor = '', c='k')
#     ax.scatter(freq,freq_avg, edgecolor = '', c='r')

     ax.plot(freq,freq_max, c='k')
     ax.plot(freq,freq_avg, c='r')

     ax.set_title('Duty Cycle')
     ax.set_xlabel('Frequency')
     ax.set_ylabel('Percentage')

     ax.axis([0,vslices,0,100])

     plt.show()




def analyse(filename,test):
    
    print('Now running test: {}'.format(test))
    dataset = pd.read_csv(filename, header=0)#, converters={0: np.int32, 1: np.int32, 2: np.int32, 3: np.int32, 4: np.int32}, dtype=np.int32)

    print(dataset.dtypes)
    #converters={'timescale': np.int32, 'frequency': np.int32, 'bandwidth': np.int32,'whitespace': np.int32, 'frame_no': np.int32}
    #cols:[0] timescale, [1] frequency, [2] bandwidth, [3] whitespace, [4] frame_no
    #in a loop (iterator), [0] = index, [1] = timescale ...
    print(dataset.columns)
    cols = dataset.columns

    
    nbins = 131072

   
    if(test == 1):
        #1. BW vs TS density plot, with TS and BW histograms/KDEs on the top and right hand side of the density plot

        bwmax = np.max(dataset[cols[2]])
        tsmax = np.max(dataset[cols[0]])

        print('BWMax: {}, TSMax: {}'.format(bwmax,tsmax))

        bwtsframe = pd.concat([dataset[cols[2]],dataset[cols[0]]], axis=1, keys=[cols[2],cols[0]])
        print(bwtsframe)

        #bwtsframe_red = bwtsframe.drop_duplicates()
        bwtsframe_red = bwtsframe.groupby([cols[2], cols[0]]).size().reset_index().rename(columns={0:'count'})
        print(bwtsframe_red)

        bwtsframe_red['ws_count'] = bwtsframe_red[cols[2]] * bwtsframe_red[cols[0]] * bwtsframe_red['count']

        print(bwtsframe_red)

        subprogram = 3
        ### PLOTTING THINGS HERE ###
        if (subprogram == 1):

            fig3 = plt.figure()
            axh = fig3.add_subplot(2,1,1)
            axh2 = fig3.add_subplot(2,1,2)

            bwtsframe[cols[2]].hist(ax=axh, bins = 200, bottom = .1, log = True)
            axh.set_yscale('log')
            axh.set_xlim(0,30000)
            axh.set_ylim(.1,1e8)
            #axh.set_xscale('log')

            bwtsframe[cols[0]].hist(ax=axh2, bins = 200, bottom = .1, log = True)
            axh2.set_yscale('log')
            axh2.set_xlim(0,60000)
            axh2.set_ylim(.1,1e8)
            #axh2.set_xscale('log')

            axh.set_title('Bandwidth Histogram')
            axh.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            axh.set_ylabel('Density')

            axh2.set_title('Duration Histogram')
            axh2.set_xlabel('Timescale (5.3ms/unit)')
            axh2.set_ylabel('Density')

            plt.show()

            ##ax1 = fig.add_subplot(1,2,1) #convention (row,col,idx)
            ##ax2 = fig.add_subplot(1,2,2)
            fig = plt.figure()
            ax = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
        #    plt.pcolormesh(mesh, norm=LogNorm(vmin=1, vmax=np.amax(mesh)), cmap='inferno')
    
            #plt.scatter(dataset[cols[2]],dataset[cols[0]])

            sc1 = ax.scatter(bwtsframe_red[cols[2]],bwtsframe_red[cols[0]],edgecolor='',c=bwtsframe_red['count'],cmap='inferno',norm=LogNorm(vmin=1,vmax=bwtsframe_red['count'].max()))
            sc2 = ax2.scatter(bwtsframe_red[cols[2]],bwtsframe_red[cols[0]],edgecolor='',c=bwtsframe_red['ws_count'],cmap='inferno',norm=LogNorm(vmin=1,vmax=bwtsframe_red['ws_count'].max()))

            #plt.hist2d(dataset[cols[2]],dataset[cols[0]],bins=1000)
            plt.colorbar(sc1,ax=ax)
            plt.colorbar(sc2,ax=ax2)

            #ticks = np.arange(0, bwmax, 6)
            #labels = range(ticks.size)
            #plt.xticks(ticks, labels)

            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylim(1,1e5)
            ax.set_xlim(10,1e5)
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_ylim(1,1e5)
            ax2.set_xlim(10,1e5)
            #ax.set_xlim(0,pxls)
            #ax.set_ylim(0,pxls)

            ax.set_title('Window Count')
            ax.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            ax.set_ylabel('Timescale (5.3ms/unit)')

            ax2.set_title('Whitespace Density')
            ax2.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            ax2.set_ylabel('Timescale (5.3ms/unit)')
    
            plt.show()

        if (subprogram == 2):
            #this second set of tests provides a plot of number of observations on the y axis versus the timescale or bandwidth. 

    #        wsvfframe = wsvfframe.groupby(['frequency','bandwidth'])['timescale'].sum().reset_index()

            bwsum = bwtsframe_red.groupby(['bandwidth'])['ws_count'].sum().reset_index()
            print('bwsum printout')
            print(bwsum)

            tssum = bwtsframe_red.groupby(['timescale'])['ws_count'].sum().reset_index()
            print('tssum printout')
            print(tssum)

        
            fig = plt.figure()
            ax = fig.add_subplot(2,2,2)
            ax2 = fig.add_subplot(2,2,4)
            ax.scatter(bwsum['bandwidth'],bwsum['ws_count'],edgecolor='')
            ax2.scatter(tssum['timescale'],tssum['ws_count'],edgecolor='')

            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlim(1,1e5)
            ax.set_ylim(1,1e9)
        
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_xlim(1,1e5)
            ax2.set_ylim(1,1e10)

            ax.set_title('Whitespace Distribution vs Bandwidth')
            ax.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            ax.set_ylabel('Unique Whitespace')

            ax2.set_title('Whitespace Distribution vs Timescale')
            ax2.set_xlabel('Timescale (5.3ms/unit)')
            ax2.set_ylabel('Unique Whitespace')

            ## window count plots - number of unique window observations

            bwhist = dataset[cols[2]]
            tshist = dataset[cols[0]]

            bwhist = bwhist.groupby(bwhist).size().reset_index().rename(columns={0:'count'})

            tshist = tshist.groupby(tshist).size().reset_index().rename(columns={0:'count'})

            print(bwhist)

            axa = fig.add_subplot(2,2,1)
            ax2a = fig.add_subplot(2,2,3)
            axa.scatter(bwhist[cols[2]],bwhist['count'],edgecolor='')
            ax2a.scatter(tshist[cols[0]],tshist['count'],edgecolor='')

            axa.set_yscale('log')
            axa.set_xscale('log')
            axa.set_xlim(1,1e5)
            axa.set_ylim(1,1e6)
        
            ax2a.set_yscale('log')
            ax2a.set_xscale('log')
            ax2a.set_xlim(1,1e5)
            ax2a.set_ylim(1,1e6)

            axa.set_title('Window Bandwidth Distribution')
            axa.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            axa.set_ylabel('Number of Observations')

            ax2a.set_title('Window Duration Distribution')
            ax2a.set_xlabel('Timescale (5.3ms/unit)')
            ax2a.set_ylabel('Number of Observations')

            plt.show()

        if (subprogram == 3):

            bwsum = bwtsframe_red.groupby(['bandwidth'])['ws_count'].sum().reset_index()
            print('bwsum printout')
            print(bwsum)

            tssum = bwtsframe_red.groupby(['timescale'])['ws_count'].sum().reset_index()
            print('tssum printout')
            print(tssum)

            numops = bwtsframe.groupby(['timescale']).size().reset_index().rename(columns={0:'count'})

            print(numops)
            

            tsmax = np.max(bwtsframe_red['timescale'])
            xmax = np.power(10,np.ceil(np.log10(tsmax)))

            ts_cumsum = np.cumsum(tssum['ws_count'][::-1])[::-1] 

            ymax = np.power(10,np.ceil(np.log10(np.max(ts_cumsum))))

            print('Partitioned Whitespace: {}'.format(ts_cumsum[0]))

            delta = np.diff(ts_cumsum)
            delta = delta*-1
            print(delta)

            fig = plt.figure()
            ax = fig.add_subplot(3,1,1)
            ax2 = fig.add_subplot(3,1,2)
            #ax.scatter(bwsum['bandwidth'],bwsum['ws_count'],edgecolor='')            
            ax.scatter(numops['timescale'],numops['count'],edgecolor='')
            ax2.scatter(tssum['timescale'],ts_cumsum,edgecolor='')
            ax2.scatter(tssum['timescale'],tssum['ws_count'],edgecolor='',color='r')
            #ax2.scatter(tssum['timescale'],ts_cumsum,edgecolor='')

            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlim(1,xmax)
            ax.set_ylim(.1,1e4)
        
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_xlim(1,xmax)
            ax2.set_ylim(1,ymax)

            #ax.set_title('Whitespace Distribution vs Bandwidth')
            #ax.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            #ax.set_ylabel('Unique Whitespace')
            
            ax.set_title('Whitespace Observation Count vs Timescale')
            ax.set_xlabel('Timescale (5.3ms/unit)')
            ax.set_ylabel('Observation Count')

            ax2.set_title('Cumulative Whitespace vs Timescale')
            ax2.set_xlabel('Timescale (5.3ms/unit)')
            ax2.set_ylabel('Unique Whitespace')

            z = np.polyfit(tssum['timescale'],tssum['ws_count'], 1)
            p = np.poly1d(z)

            ax2.plot(tssum['timescale'],p(tssum['timescale']), 'k--')

            z2 = np.polyfit(np.log(tssum['timescale']),tssum['ws_count'], 2)
            p2 = np.poly1d(z2)
            ax2.plot(tssum['timescale'],p2(np.log(tssum['timescale'])), 'g--')

            ax3 = fig.add_subplot(3,1,3)

            perh_cumsum = (ts_cumsum/np.max(ts_cumsum)) *100
            ax3.scatter(tssum['timescale'],perh_cumsum, edgecolor='')
            ax3.set_xscale('log')
            ax3.set_xlim(1,xmax)

            z3 = np.polyfit(np.log(tssum['timescale']),perh_cumsum, 3)
            p3 = np.poly1d(z3)

            ax3.plot(tssum['timescale'],p3(np.log(tssum['timescale'])), 'k--')
            ## window count plots - number of unique window observations
            plt.show()

        #elif(test == 8):
        #BW vs WS and TS vs WS analysis - Test 1 is the complementary analysis of this test
        #we want to plot both unique whitespace incurred as well as total whitespace as the x axis decreases

        

    elif(test == 2):
    #2. WS vs Bins (real frequency - may need to preserve band starting frequency ....) 
    #Series 1 - WS per bin due to partitioning algorithm
    #Series 2 - WS per bin based on the unpartitioned spectrum
        
        subprogram = 3

        wsvfframe = pd.concat([dataset[cols[1]],dataset[cols[2]],dataset[cols[0]]], axis=1, keys=[cols[1],cols[2],cols[0]])

        #wfc:[0] frequency, [1] bandwidth, [2] timescale
        #in a loop (iterator), [0] = index, [1] = frequency ...
        wfc = wsvfframe.columns
        print(wfc)
        
        freq = np.linspace(0,nbins-1,nbins)

        print(freq)
        relws = np.zeros(nbins)
        compws = np.zeros(nbins)
        diffarry = np.zeros(nbins)
       
        wsvfframe = wsvfframe.groupby(['frequency','bandwidth'])['timescale'].sum().reset_index()

        for row in wsvfframe.itertuples():
            relws[row[1]:row[1]+row[2]] += row[3]

            
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(2,1,1)
            axa = fig.add_subplot(2,1,2)
        
            ax.scatter(freq,relws,edgecolor='')

            relwsS = np.sort(relws)

            axa.scatter(freq,relwsS[::-1],edgecolor='')

            axa.set_title('Whitespace Frequency Distribution CCDF')
            axa.set_xlabel('Descending Magnitude Ordered Bin Count')
            axa.set_ylabel('Whitespace Units over Observation Period')

            ax.set_title('Whitespace Frequency Distribution')
            ax.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            ax.set_ylabel('Whitespace Units')

            plt.show()

        ### comaprison dataset NOTE: BW must = 1 for this dataset to be valid ###

        print('Select comparison windowset. ENSURE BW = 1')
        bw_1_file = filedialog.askopenfilename()

        bw_1_dataset = pd.read_csv(bw_1_file, header=0, dtype=np.int32)
        bcols = bw_1_dataset.columns

        bw_1_dataset = bw_1_dataset.groupby(by=['frequency'])['timescale'].sum().reset_index()

        for row in bw_1_dataset.itertuples():
            compws[row[1]] = row[2]
           
        #### test zone ####

        ### Profile the thing :)
        #pr = cProfile.Profile()
        #pr.enable()
        ### ... do something ...
        #pr.disable()
        #s = io.StringIO()
        #sortby = 'cumulative'
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print(s.getvalue())   

        ##### mmm tests ####

        diffarry = np.subtract(compws,relws)

        dfcomp = pd.DataFrame(data={'rel': relws, 'comp': compws})
        
        print(dfcomp)
        dfcomp = dfcomp.sort_values(['comp','rel'],ascending=False)

        print(dfcomp)

        d2comp = np.array([compws,relws])
        d2comp[d2comp[:,0].argsort()]

        if False:
            figz = plt.figure()
        
            axz = figz.add_subplot(2,1,1)
            axz.scatter(freq,dfcomp['comp'],edgecolor='',label='Absolute Spectrum')
            axz.scatter(freq,dfcomp['rel'],edgecolor='',c='r',label='Windowed Spectrum')
            axz.legend()

            dfcomp = dfcomp.sort_values(['rel','comp'],ascending=False)

            axza = figz.add_subplot(2,1,2)
            axza.scatter(freq,dfcomp['comp'],edgecolor='',label='Absolute Spectrum')
            axza.scatter(freq,dfcomp['rel'],edgecolor='',c='r',label='Windowed Spectrum')
            axza.legend()

            plt.show()

        if False:

            fig = plt.figure()
            ax = fig.add_subplot(3,2,1)
            ax2 = fig.add_subplot(3,2,3)
            ax3 = fig.add_subplot(3,2,5)

            axa = fig.add_subplot(3,2,2)
            ax2a = fig.add_subplot(3,2,4)
            ax3a = fig.add_subplot(3,2,6)
        #2.1 Companion plot to WS vs Bins, - % covered vs Bins, showing the amount of real spectrum covered by the partitioned algo as a percentage

            ax.scatter(freq,relws,edgecolor='')
            ax2.scatter(freq,diffarry,edgecolor='')
    #        ax3.scatter(freq,compws,edgecolor='',label='Absolute Spectrum')
            ax3.scatter(bw_1_dataset['frequency'],bw_1_dataset['timescale'],edgecolor='',label='Absolute Spectrum')
            ax3.scatter(freq,relws,edgecolor='',c='r',label='Windowed Spectrum')

            #CDFs?

            relwsS = np.sort(relws)
            diffarryS = np.sort(diffarry)
            compwsS = np.sort(compws)     

    #        compwspad_r = np.resize(compwspad,(1,131072))
    #        compwsS = np.sort(compwspad_r)
    #        relwsS = np.sort(relws)

            axa.scatter(freq,relwsS[::-1],edgecolor='')
            axa.set_title('Whitespace Frequency Distribution CCDF')
            axa.set_xlabel('Descending Magnitude Ordered Bin Count')
            axa.set_ylabel('Whitespace Units over Observation Period')
            ax2a.scatter(freq,diffarryS[::-1],edgecolor='')
            ax2a.set_title('Whitespace Frequency Distribution Difference CCDF')
            ax2a.set_xlabel('Descending Magnitude Ordered Bin Count')
            ax2a.set_ylabel('Whitespace Units Difference over Observation Period')
        
            ax3a.set_title('Whitespace Frequency Distribution CCDF')
            ax3a.set_xlabel('Descending Magnitude Ordered Bin Count')
            ax3a.set_ylabel('Whitespace Units over Observation Period')
            ax3a.scatter(freq,compwsS[::-1],edgecolor='',label='Absolute Spectrum')
        
    #        ax3a.scatter(freq,compwsS[::-1],edgecolor='',label='Absolute Spectrum')
            ax3a.scatter(freq,relwsS[::-1],edgecolor='',c='r',label='Windowed Spectrum')
            ax3a.legend()

           # ax.set_yscale('log')
           # ax.set_xscale('log')
           # ax2.set_yscale('log')
           # ax2.set_xscale('log')

            ax.set_title('Whitespace Frequency Distribution')
            ax.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            ax.set_ylabel('Whitespace Units')

            ax2.set_title('Missed Window Coverage (Absolute minus Windowed)')
            ax2.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            ax2.set_ylabel('Number of Whitespace Units Difference')

            ax3.set_title('Window Coverage Comparison')
            ax3.set_xlabel('Bandwidth in Bins (190Hz/bin)')
            ax3.set_ylabel('Whitespace Units')
            ax3.legend()
            plt.show()




    elif(test == 3):

    #3. Instantaneous bandwidth (total) vs Time (real - in frames) 
    #Possible second series showing the number of instantaneous windows/fragmentation - fragmentation is probably more interesting
   
        #cols:[0] timescale, [1] frequency, [2] bandwidth, [3] whitespace, [4] frame_no
        bwvt = pd.concat([dataset[cols[2]],dataset[cols[0]],dataset[cols[4]]], axis=1, keys=[cols[2],cols[0],cols[4]])

        #bcols:[0] bandwidth, [1] timescale, [2] frame_no
        bcols = bwvt.columns

        maxframe = np.max(bwvt[bcols[2]])
        bwspread = np.zeros(maxframe)

        time = np.linspace(0,maxframe-1,maxframe)
        print(time)

        for row in bwvt.itertuples():
            bwspread[row[3]-1-row[2]:row[3]] += row[1]

        #bwspread[maxframe-1] = 0 #clear the final value as it gives an improper indication of continuous bandwidths

        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax.scatter(time,bwspread,edgecolor='')

        #ax.set_yscale('log')
        #ax.set_xscale('log')

        ax.set_title('Bandwidth over Time')
        ax.set_xlabel('Continuous Time (5.3ms/unit)')
        ax.set_ylabel('Aggregate Instantaneous Bandwidth (190Hz/bin)')
        
        plt.show()

    #4. Average window duration vs Time - Showing the average window persistance (in frames) with respect to time (in frames) - this may be confusing?

    elif(test == 4):

        #cols:[0] timescale, [1] frequency, [2] bandwidth, [3] whitespace, [4] frame_no
        dvt = pd.concat([dataset[cols[0]],dataset[cols[4]]], axis=1, keys=[cols[0],cols[4]])

        #dcols:[0] timescale, [1] frame_no
        dcols = dvt.columns

        maxframe = np.max(dvt[dcols[1]])
        tavgspread = np.zeros(maxframe)
        ttotalspread = np.zeros(maxframe)

        time = np.linspace(0,maxframe-1,maxframe)

        wincount = np.zeros(maxframe)
        
        print(dvt)

        currframe = 0
        count = 1

        for row in dvt.itertuples():
            if row[2] > currframe:
                    
                tavgspread[currframe-1] /= count

                wincount[currframe] = count
                count=1
                currframe = row[2]
            else: count += 1

            tavgspread[row[2]-1] += row[1]
            ttotalspread[row[2]-1] += row[1]

        tavgspread[maxframe-1] /= count #clear the final value as it gives an improper indication of continuous bandwidths
        ttotalspread[maxframe-1] = 0
        wincount[maxframe-1] = 0

        fig = plt.figure()
        ax = fig.add_subplot(3,2,1)
        ax2 = fig.add_subplot(3,2,3)
        ax3 = fig.add_subplot(3,2,5)

        axa = fig.add_subplot(3,2,2)
        ax2a = fig.add_subplot(3,2,4)
        ax3a = fig.add_subplot(3,2,6)

        ax.scatter(time,tavgspread,edgecolor='')
        ax2.scatter(time,ttotalspread,edgecolor='')
        ax3.scatter(time,wincount,edgecolor='')

        #ax.set_yscale('log')
        #ax.set_xscale('log')

        ax.set_title('Window Duration over Time')
        ax.set_xlabel('Continuous Time (5.3ms/unit)')
        ax.set_ylabel('Average Window Duration (5.3ms/unit)')

        ax2.set_title('Aggregate Window Duration over Time')
        ax2.set_xlabel('Continuous Time (5.3ms/unit)')
        ax2.set_ylabel('Total Aggregate Window Duration (5.3ms/unit)')

        ax3.set_title('Total Windows over Time')
        ax3.set_xlabel('Continuous Time (5.3ms/unit)')
        ax3.set_ylabel('Number of Windows')
        
        tavgsS = np.sort(tavgspread)
        ttotsS = np.sort(ttotalspread)
        wincS = np.sort(wincount)
        
        axa.scatter(time,tavgsS[::-1],edgecolor='')
        axa.set_title('Window Duration over Time CCDF')
        axa.set_xlabel('Descending Magnitude Observation Index')
        axa.set_ylabel('Average Window Duration (5.3ms/unit)')

        ax2a.scatter(time,ttotsS[::-1],edgecolor='')
        ax2a.set_title('Aggregate Window Duration over Time CCDF')
        ax2a.set_xlabel('Descending Magnitude Observation Index')
        ax2a.set_ylabel('Total Aggregate Window Duration (5.3ms/unit)')
        
        ax3a.set_title('Total Windows over Time CCDF')
        ax3a.set_xlabel('Descending Magnitude Observation Index')
        ax3a.set_ylabel('Window Count')
        ax3a.scatter(time,wincS[::-1],edgecolor='')

        plt.show()




    if (test == 5):

    #5. Quality vs Bins - quality computed using weighting algorithm and assigning those weights to the relative bin(s) spanned by the appropriate window. - Sum the total weights, and/or average weights and display those?

        
        #cols:[0] timescale, [1] frequency, [2] bandwidth, [3] whitespace, [4] frame_no
        qvf = pd.concat([dataset[cols[0]],dataset[cols[2]],dataset[cols[1]]], axis=1, keys=[cols[0],cols[2],cols[1]])

        #qcols: [0] timescale, [1] bandwidth, [2] frequency
        qcols = qvf.columns

        freq = np.linspace(0,nbins-1,nbins)
        qualarry = np.zeros(nbins)

        tmin = 10
        bmin = 66 #These should already be adhered to

        for row in qvf.itertuples():
            qual = (row[1]-tmin)/tmin * (row[2]/bmin)
            if qual > 1:
                qualarry[row[3]:row[3]+row[2]-1] += qual

        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax.scatter(freq,qualarry,edgecolor='')

        qualarryord = np.sort(qualarry)
#        qualarryord = qualarryord[::-1]
        ax2 = fig.add_subplot(2,1,2)
        ax2.scatter(freq,qualarryord[::-1],edgecolor='')

        #ax.set_yscale('log')
        #ax.set_xscale('log')

        ax.set_title('Quality per Bin over Observation Period')
        ax.set_xlabel('Frquency in Bins (190Hz/bin)')
        ax.set_ylabel('Total Bin Quality')

        ax2.set_title('Quality per Bin over Observation Period Ordered')
        ax2.set_xlabel('Count (Eventually as a percentage)')
        ax2.set_ylabel('Total Bin Quality')

        plt.show()

def legacy_sns (filename):

    wins = pd.read_csv(filename, header=0) #, chunksize = chnk)
    print(wins.columns)

    #cols:timescale,frequency,bandwidth,whitespace,frame_no
    cols = wins.columns
    
    #print(wins.head(10))
    #ts = wins.sort_values([cols[0],cols[2]])
 
    #uniq = pd.unique(wins[cols[0]].values.ravel())#, columns = [cols[0],cols[3]])
    #ts_sort = np.c_[uniq,np.zeros(uniq.size)]    
    #ts_sum = np.empty_like(ts_sort)

    ##Row[0] is the index given by the dataframe
    ##Here the unique TS is used as the index, where the unique whitespace value is summed for each TS 
    ##TS is already sorted ascending, by nature of the detection
    ## THIS WILL ONLY WORK IF EVERY TS IS OBSERVED !!!BAD!!!
    #for row in wins.itertuples():
    #    ts_sort[row[1]-1,1] += row[4]

    
    sns.jointplot(wins[cols[2]],wins[cols[0]],kind="hex",color="#4CB391", stat_func=None)
    sns.jointplot(x='bandwidth',y='timescale',data = wins,kind="kde",color="#4CB391", stat_func=None)

    plt.savefig("heatmap.png")
    plt.show()

def legacy(filename):

    wins = pd.read_csv(filename, header=0) #, chunksize = chnk)
    print(wins.columns)

    #cols:timescale,frequency,bandwidth,whitespace,frame_no
    cols = wins.columns
    
    #print(wins.head(10))
    #ts = wins.sort_values([cols[0],cols[2]])
 
    #uniq = pd.unique(wins[cols[0]].values.ravel())#, columns = [cols[0],cols[3]])
    #ts_sort = np.c_[uniq,np.zeros(uniq.size)]    
    #ts_sum = np.empty_like(ts_sort)

    ##Row[0] is the index given by the dataframe
    ##Here the unique TS is used as the index, where the unique whitespace value is summed for each TS 
    ##TS is already sorted ascending, by nature of the detection
    ## THIS WILL ONLY WORK IF EVERY TS IS OBSERVED !!!BAD!!!
    #for row in wins.itertuples():
    #    ts_sort[row[1]-1,1] += row[4]

    
    #sns.jointplot(wins[cols[2]],wins[cols[0]],kind="hex",color="#4CB391", stat_func=None)
    #sns.jointplot(x='bandwidth',y='timescale',data = wins,kind="kde",color="#4CB391", stat_func=None)
    #plt.show()

    x = max(wins[cols[2]])
    y = max(wins[cols[0]])
    fn = max(wins[cols[4]])

    nbins = 131072

    print('Max Bandwidth: {}  Max Timescale: {}'.format(x,y))

    mesh = np.zeros((x,y))
    num_frames = np.zeros(fn)

    freqdensity = np.zeros(nbins)

    #print(mesh)

    uniq = pd.unique(wins[cols[2]].values.ravel())
    uniq.sort()

    ws_sort = np.c_[uniq,np.zeros(uniq.size)]    
    ws_sum = np.empty_like(ws_sort)

    count = 0
    totalws = 0
    #Row[0] is the index given by the dataframe
    for row in wins.itertuples():
        mesh[row[3]-1,row[1]-1] += row[4] 
        #temp = np.where(ws_sort[:,0]==row[3])
        #ws_sort[temp,1] += row[4]
        totalws += row[4]
    ##     mesh[row[1]-1,row[3]-1] += row[4]
        
        #populate frequency density array
        #for i in range(row[2],row[2]+row[3]-1):
        #    freqdensity[i]+=row[1] 

        count += 1
        if (count%100000 == 0): print('Up to: {}'.format(count))

    print('Total Whitespace: ')
    print(totalws)

    fig = plt.figure()
    ##ax1 = fig.add_subplot(1,2,1) #convention (row,col,idx)
    ##ax2 = fig.add_subplot(1,2,2)

    ##pd.DataFrame.hist(wins,column=cols[0],bins=(x+1),log=True,ax=ax1)
    ##ax1.set_xlim([0,50])
    ##ax1.set_title('Timescale Density')
    ##ax1.set_xlabel('Timescale')
    ##ax1.set_ylabel('# Observations')

    ##ax2.plot(num_frames)
    ##ax2.set_ylim([0,100])
    ##ax2.set_title('Whitespace Channels')
    ##ax2.set_xlabel('Frame Number')
    ##ax2.set_ylabel('# Whitespace Channels')
    ##plt.show()

    #print('Mesh populated')
    #print(mesh)
 
    ax = fig.add_subplot(1,1,1)
    plt.pcolormesh(mesh, norm=LogNorm(vmin=1, vmax=np.amax(mesh)))
    plt.colorbar()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlim(1,x)
    ax.set_ylim(1,y)
    ax.set_title('Whitespace Density')
    ax.set_xlabel('Bandwidth in Bins (190Hz/bin)')
    ax.set_ylabel('Timescale (5.3ms/unit)')

    #ax1 = fig.add_subplot(2,1,1)
    #ax1.plot(freqdensity)
    ##ax1.set_xlim([0,50])
    #ax1.set_title('Whitespace Density per Bin')
    #ax1.set_xlabel('Frequency in Bins (190Hz/bin)')
    #ax1.set_ylabel('Whitespace')

    plt.show()

    #ts_sum = ts_sort
    # np.copyto(ts_sum,ts_sort)
    np.copyto(ws_sum,ws_sort)
    
    #df = pd.DataFrame(data=ts_sort,columns=[cols[0],cols[3]])
    #print(df)

    #for i in range(uniq.size-1, 0, -1):
    #    ts_sum[i-1,1] += ts_sum[i,1]
        #print(ts_sum[i,1])
    for i in range(uniq.size-1, 0, -1):
        ws_sum[i-1,1] += ws_sum[i,1]

    print('\nws_sort:')
    print(ws_sort)
    print('\nws_sum:')
    print(ws_sum)

    sns.jointplot(ws_sort[:,0],ws_sort[:,1],kind="hex",color="#4CB391")
    plt.show()

    #for i in range(len(wins)):
     #       set1 = np.array(wins[wins.columns[0::2]])[i]

    #wins.plot(x=cols[0],y=cols[3],kind='scatter')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.plot(ts_sort)
    #ax.set_yscale('log')
    #plt.semilogy(ts_sort[0], ts_sort[1], ts_sum[0], ts_sum[1])
    ax.semilogy(ws_sort[:,0], ws_sort[:,1], label='Unique Whitespace')
    ax.semilogy(ws_sum[:,0], ws_sum[:,1], label='Cumulative Whitespace')

    ax.set_yscale('symlog') #super important to plot '0' values
    ax.legend()
    plt.show()




#BEGIN THE SIMULATOR - Should put this in a separate file, but visualstudio is mean when it comes to selecting the file to execute etc

class Device:
    def __init__(self, ts, bw, id):
        self.ts = ts
        self.bw = bw
        self.id = id
        
        self.freq = 0

        self.run_time = 50 #this number should be modified based on access model

        self.sched_time = 0

        self.total_sched_time = 0 #total running time
        self.last_scheduled = 0 
        self.waiting_time = 0 #total waiting time IF communication interrupted
        #maybe need a scheduling flag?

    #FIXME
    def schedule(self, freq, bandwidth, startframe):
        if bandwidth < self.bw:
            print('DEVICE {} SCHEDULING ERROR, BW TOO SMALL'.format(self.id))
        else: 
            self.freq = freq
            self.waiting_time = self.waiting_time + (startframe - (self.last_scheduled - 1))
            self.sched_time = startframe

    def deschedule(self, endframe):
        if endframe - self.sched_time < self.run_time:
            print("device {} only ran for {}. Required runtime: {}".format(self.id,endframe-self.sched_time,self.run_time))

        self.total_sched_time = self.total_sched_time + (endframe - self.sched_time) 
        self.last_scheduled = endframe

    def totalTime(self):
        return self.total_sched_time

    def totalWait(self):
        return self.waiting_time

    def getFreq(self):
        return self.freq

    def getBW(self):
        return self.bw

    def getID(self):
        return self.id

    def getRemianingRun(self, frame):
        return self.run_time - (frame - self.sched_time)

    #will not implement getTS yet, as we have no true concept of TS as everything is running without a-priori knowledge

#class DeviceCreator

class Slice: #has been supersceeded by simply using a dataframe
    def __init__(self,frequency,bandwidth):
        self.freq = frequency
        self.bw = bandwidth
    
    def getFreq(self):
        return self.freq

    def getBW(self):
        return self.bw

class SpectrumEntity:
    def __init__(self,filename):
        self.framecount = 0
        self.dataset = pd.read_csv(filename, header=0)#, converters={0: np.int32, 1: np.int32, 2: np.int32, 3: np.int32, 4: np.int32}, dtype=np.int32)

        self.framemax = np.max(self.dataset['frame_no'])
        self.dataset = self.dataset.sort_values(['frequency'], ascending=True)
        self.dataset['start'] = self.dataset['frame_no'].subtract(self.dataset['timescale']-1)

        self.state = self.dataset.groupby(self.dataset['start'])
        #self.stateGroups = dict(list(self.state))

        self.spect_df = pd.DataFrame()
        #self.sliceList = []

        self.sliceFrame = pd.DataFrame()

        #when nextFrame is called check that result is not null
    def getNextFrame(self):

        if self.framecount == self.framemax:
            print("what we gon do now?")
            return null

        #for i in range(13):
        #clear current buffer
        #del self.sliceList[:]
        self.framecount = self.framecount + 1

        #create frame list of slices
            
    #       for key, group in self.state:
    #           if key < 13:
    #               print(key)
    #               #print(group)
    #               self.spect_df = self.spect_df.append(group)
    #               self.spect_df = self.spect_df[self.spect_df.frame_no > key] #this is pretty inefficient
    #               print(self.spect_df) #this should be a self contained dataframe of the current windows, just take a slice from this dataset and gg

        if self.framecount in self.state.groups:
            self.spect_df = self.spect_df.append(self.state.get_group(self.framecount))
            self.spect_df = self.spect_df[self.spect_df.frame_no > self.framecount]
            #self.dataset = self.dataset.sort_values(['frequency'], ascending=True)
            self.spect_df = self.spect_df.sort_values(['frequency'], ascending=True)
        else: 
            self.spect_df = self.spect_df[self.spect_df.frame_no > self.framecount] #this is pretty inefficient

        #slice spect_df
        #print(self.spect_df)
        self.sliceFrame = self.spect_df[['frequency','bandwidth']].reset_index(drop=True)
        
        if False:                    
            print(self.framecount)
            print(self.spect_df)
            print(self.sliceFrame)

        #check for adjacency - maybe later, currently ignore adjacency


        #slices should be sorted as the generation of them is procedural
        #return self.sliceList
        return self.sliceFrame
        
class DeviceQueue:
    def __init__(self,numDev):
        self.queue = []

        #create the devices to place in the queue
        #ts and bw could be generated uniquely for each device, currently we are using a placeholder
        self.ts = 11
        self.bw = 66
        
        for i in range(numDev):
            self.queue.append(Device(self.ts,self.bw,i))

        print("{} devices created".format(len(self.queue)))

        #this is called by the scheduler
    def push(self,dev,frame):
        #push takes in a device object and adds it to the queue
        dev.deschedule(frame)
        self.queue.append(dev) 

    def peek(self):
        if not self.queue:
            return null
        else:
            return self.queue[0]

    def search(self,bw,time):
        for dev in self.queue:
            if dev.getBW <= bw:
                temp = dev
                self.queue.remove(dev)
                return temp
                break

    def pop(self):
        if not self.queue:
            return null
        else:
            return self.queue.popleft()

    def len(self):
        return len(self.queue)

class Scheduler:
    def __init__(self,filename,numDev):
        #create spectrum entity
        #create UDQ
        #create devices (inform queue of number of devices to populate)
        
        #implement logging
     
        self.RSL = [] #remainingSpectrumList
        self.ADL = [] #allocatedDeviceList

        self.currentFrame = 1

        #devices created within queue object
        #self.UDQ = DeviceQueue(numDev) #working around the queue as it does not seem efficient

        self.UDL = [] #unallocatedDeviceList
        self.ts = 11
        self.bw = 66
        
        for i in range(numDev):
            self.UDL.append(Device(self.ts,self.bw,i))

        print("{} devices created".format(len(self.UDL)))

        #slices prepared within SE object
        self.SE = SpectrumEntity(filename)
        
        self.run_list()

        #implement operational loop within here
    def run(self):
        #check SE for spectrumstate
        self.spectrumState = self.SE.getNextFrame()

        #print(self.spectrumState)

        adevs = len(self.ADL)
        adevIdx = 0

        udevs = len(self.UDL)
        udevIdx = 0

        remainBW = 0
        self.tempADL = self.ADL.copy()

        uFlag = False
        noAlloc = False

        #compare spectrum state with ADL to ensure allocations still valid
        while not self.spectrumState.empty:
            #print('Frame number: {}'.format(self.currentFrame))

            if self.ADL:
                for row in self.spectrumState.itertuples(): #iterate through all of the available spectrum
                    #row['frequency'], row['bandwidth']

                    uBW = row['bandwidth']
                    uFreq = row['frequency'] #beginnign of the unallocated frequency for that particular slice

                    for adev in range(adevIdx,len(self.ADL)): #FIX ALL THESE TO HAVE CORRECT INDEXATION!!!!!!!!!!!

                        #still need to take into account duration expired cases
                    
                        #check frequency and bandwidth 
                        
                        #if we are in to the slice and the previous slice was not befitting
                        if adev.getFreq + adev.getBW < uFreq and uFlag:
                            #device needs to be deallocated
                            uFlag = False
                            self.UDL.append(adev)
                            self.tempADL.remove(adev)

                            adevIdx = adevIdx + 1
                            continue 

                        if adev.getFreq + adev.getBW <= row['frequency'] + row['bandwidth'] and adev.getFreq >= row['frequency']:
                            #device exists within slice

                            #this does not work correctly for fractioned slices methinks.
                            if uFreq != adev.getFreq:
                                print('Gap in allocation detected. Dev start: {}, window start {}'.format(adev.getFreq,uFreq))
                                self.RSL.append([uFreq,uFreq - dev.getFreq], columns=['frequency','bandwidth'])
                                uBW = uBW - (uFreq - dev.getFreq)
                            
                            uFreq = adev.getFreq + adev.getBW #this will be one position after the end of the allocation, i.e. the next position not conflicting with the current device
                            uBW = uBW - adev.getBW

                            if uBW < 0:
                                print('Well, we have a negative bandwidth here, thats not good ....')
                            elif uBW == 0:
                                print('Slice completely allocated')
                                adevIdx = adevIdx + 1
                                continue
                            else: 
                                print('Allocation valid')
                        
                        elif dev.getFreq >= row['frequency'] + row['bandwidth']:
                            #device does not exist in this slice, thus remaining slice is free
                            print('Device {} does not exist in this slice, remaining slice: {} - validate with uBW: {}'.format(adevIdx,row['frequency']+row['bandwidth']-uFreq,uBW))

                            adevIdx = adevIdx #this device still needs to be checked

                            self.RSL.append([uFreq,uBW], columns=['frequency','bandwidth'])
                            uFlag = True;
                        
                            break #currently allocated devices do not exist in this slice


                        #other devices may also not exist within this bound ... probably should not break then ..
                        #not if they are sorted.

                        adevIdx = adevIdx + 1
                        #update RSL

                        #deallocate conflicting devices
            
                    #for udev in range(self.UDQ.len):
            else: 
                noAlloc = True

            print('Allocated devices checked: {}. Devices removed: {}'.format(len(self.ADL), len(self.ADL) - len(self.tempADL)))
            self.ADL = self.tempADL.copy() #I REALLY dislike this, however its annoying to remove elements from a list as you iterate over it

            if self.UDL:
                #unallocated devices, interrogate unallocated list then find potential slots for each device waiting
                if noAlloc:
                    self.RSL = self.spectrumState.copy()

                for udev in range(len(self.UDL)):
                    print(udev)
                    #to schedule device increment the frequency by the devices bandwidth and subtract the available bandwidth for that opportunity
                    print(self.RSL[self.RSL['frequency'] >= self.UDL[udev].getBW()].iloc[0:2,0:2]) 
                    print(self.RSL[self.RSL['frequency'] >= self.UDL[udev].getBW()].iloc[0:2,0:2].values.tolist()) 


            #generate RSL

            #check RSL against UDQ and update RSL accordingly

            #perform logging on current state

            #sort ADL by frequency to reduce number of comparisons

            #loop again until we have exhausted the spectrum
            self.currentFrame = self.currentFrame + 1
            self.spectrumState = self.SE.getNextFrame()

            if self.spectrumState.empty:
                print("End of spectrum reached!")
                break

        print("should output things here, plots etc")
        #finish up
        print("scheduler terminating")

    def run_list(self):
        #check SE for spectrumstate
        self.spectrumState = self.SE.getNextFrame().values.tolist()

        #print(self.spectrumState)
        remainBW = 0
        adevIdx = 0

        noAlloc = False

        removed = 0
        #compare spectrum state with ADL to ensure allocations still valid
        while self.spectrumState:
            #print('Frame number: {}'.format(self.currentFrame))

            
            if self.ADL:
                i = 0
                for row in self.spectrumState: #iterate through all of the available spectrum
                    #row['frequency'], row['bandwidth']

                    uFreq = row[0] #beginnign of the unallocated frequency for that particular slice
                    uBW = row[1]
                    
                    while i < len(self.ADL):
                    #for adev in islice(self.ADL,adevIdx,None): #FIX ALL THESE TO HAVE CORRECT INDEXATION!!!!!!!!!!!

                        #check frequency and bandwidth                       
                        #if we are in to the slice and the previous slice was not befitting
                        if self.ADL[i].getFreq() + self.ADL[i].getBW() < uFreq:
                            #device needs to be descheduled
                            print('Deallocating: dev {} from [{},{}]'.format(self.ADL[i].getID(),self.ADL[i].getFreq(),self.ADL[i].getBW()))
                            self.ADL[i].deschedule(self.currentFrame)
                            self.UDL.append(self.ADL[i])
                            del self.ADL[i]

                            #i = i - 1
                            removed = removed + 1
                            continue 

                        uFlag = False

                        #still need to take into account duration expired cases
                        if self.ADL[i].getRemianingRun(self.currentFrame) <= 0:
                            #device needs to be descheduled
                            print('Runtime expired: dev {} from [{},{}]'.format(self.ADL[i].getID(),self.ADL[i].getFreq(),self.ADL[i].getBW()))
                            self.ADL[i].deschedule(self.currentFrame)
                            self.UDL.append(self.ADL[i])
                            del self.ADL[i]

                            #i = i - 1
                            removed = removed + 1
                            continue 

                        elif self.ADL[i].getFreq() >= row[0] + row[1]:
                            #device does not exist in this slice, thus remaining slice is free
                            #print('Device {} does not exist in this slice, remaining bandwidth in slice: {} - validate with uBW: {}'.format(adev.getID(),row[0]+row[1]-uFreq,uBW))

                            self.RSL.append([uFreq,uBW])

                            #this device still needs to be checked
                            break #currently allocated devices do not exist in this slice

                        elif self.ADL[i].getFreq() + self.ADL[i].getBW() <= row[0] + row[1] and self.ADL[i].getFreq() >= row[0]:
                            #device exists within slice

                            #this does not work correctly for fractioned slices methinks.
                            if uFreq != self.ADL[i].getFreq():
                                print('Gap in allocation detected. Dev start: {}, window start {}'.format(self.ADL[i].getFreq(),uFreq))
                                self.RSL.append([uFreq,uFreq - self.ADL[i].getFreq()])
                                uBW = uBW - (uFreq - self.ADL[i].getFreq())
                            
                            uFreq = self.ADL[i].getFreq() + self.ADL[i].getBW() #this will be one position after the end of the allocation, i.e. the next position not conflicting with the current device
                            uBW = uBW - self.ADL[i].getBW()

                            if uBW < 0:
                                print('Well, we have a negative bandwidth here, thats not good ....')
                            elif uBW == 0:
                                print('Slice completely allocated!')
                            #else: 
                                #print('Allocation valid')

                        #end of loop and device is valid, increment i
                        i = i + 1

                    #remaining spectrum is unoccupied
                    if i == len(self.ADL):
                        self.RSL.append([row[0],row[1]])

                    #for udev in range(self.UDQ.len):

                #sort newly populated RSL
                adevIdx = 0
                #print('Remaining Spectrum List')
                #print(self.RSL)
                self.RSL.sort(key=lambda x: x[0])
                #print(self.RSL)
            else: 
                noAlloc = True

            allocated = 0
            if self.UDL:
                #unallocated devices, interrogate unallocated list then find potential slots for each device waiting
                if noAlloc:
                    self.RSL = self.spectrumState.copy()
                    noAlloc = False

                #for udev in self.UDL:
                #    iterations = iterations + 1
                #    #to schedule device increment the frequency by the devices bandwidth and subtract the available bandwidth for that opportunity
                #    #print(self.RSL[self.RSL['frequency'] >= self.UDL[udev].getBW()].iloc[0:2,0:2]) 
                #    #print(self.RSL[self.RSL['frequency'] >= self.UDL[udev].getBW()].iloc[0:2,0:2].values.tolist()) 
                #    for row in self.RSL:
                #        if row[1] >= udev.getBW():
                #            #print(row)
                #            udev.schedule(row[0],row[1],self.currentFrame)
                #            row[1] = row[1] - udev.getBW()
                #            row[0] = row[0] + udev.getBW()
                #            self.ADL.append(udev)
                #            self.UDL.remove(udev) #wont work
                #            break   

                i = 0
                offset = 0
                while i < len(self.UDL):
                    #to schedule device increment the frequency by the devices bandwidth and subtract the available bandwidth for that opportunity
                    
                    #this looks very C and not at all pythonic :)
                    for row in self.RSL:
                        if row[1] >= self.UDL[i].getBW():
                            #print(i)
                            #print(offset
                            print('Allocating {} to [{},{}]'.format(self.UDL[i].getID(),row[0],row[1]))
                            self.UDL[i].schedule(row[0],row[1],self.currentFrame)                            
                            row[0] = row[0] + self.UDL[i].getBW()
                            row[1] = row[1] - self.UDL[i].getBW()
                            self.ADL.append(self.UDL[i])
                            del self.UDL[i] #might work
                            #offset = offset + 1
                            i = i - 1
                            allocated = allocated + 1
                            break   
                    i = i + 1
                
                #print('Iterations of self.UDL: {}, self.UDL size: {}'.format(i,len(self.UDL)))
                print('Original spectrum:')
                print(self.spectrumState)
                print('Remaining spectrum for frame {}:'.format(self.currentFrame)) 
                print(self.RSL)

            #generate RSL

            #check RSL against UDQ and update RSL accordingly

            #perform logging on current state

            #sort ADL by frequency to reduce number of comparisons

            #loop again until we have exhausted the spectrum
            self.currentFrame = self.currentFrame + 1
            self.spectrumState = self.SE.getNextFrame().values.tolist()

            print('Frame {} - Allocated devices checked: {}. Devices removed: {}. Devices reallocated: {}. Devices unallocated: {}'.format(self.currentFrame-1,len(self.ADL), removed, allocated, len(self.UDL)))
            removed = 0
            del self.RSL[:]

            if not self.spectrumState:
                print("End of spectrum reached!")
                break

        print("should output things here, plots etc")
        #finish up
        print("scheduler terminating")
        


        


def dev_sim(filename):

    print('Now running Device Simulation')


    runtest = Scheduler(filename,200)

    print('Device simulation ended')
    dataset = pd.read_csv(filename, header=0)#, converters={0: np.int32, 1: np.int32, 2: np.int32, 3: np.int32, 4: np.int32}, dtype=np.int32)

    #print(dataset.dtypes)

    #cols:[0] timescale, [1] frequency, [2] bandwidth, [3] whitespace, [4] frame_no
    #in a loop (iterator), [0] = index, [1] = timescale ...
    print(dataset.columns)
    cols = dataset.columns
    
    nbins = 131072

    #This set of tests are designed to simulate channel occupancy/utilisation of secondary devces accessing the whitespace spectrum, given particular access requirements.
    #These requirements are (initially): Bandwidth, minimum timescale.
    #The tests will be performed accross the number of devices operating and the resulting utilisation of the spectrum
    #Note that complex scheduling algorithms will not be explored (at least initially), as this is an extremely deep topic within itself. 

    #the resulting plots will be: Spectrum utilisation/total throughput versus number of devices, with a family of curves, each curve will either be a variance on device required timescale or bandwidth. 
    #a comparison plot for this will also be generated detailing the per device throughput and how it decays as a function of device density.

    #it is also of note that these models do not take into account channel capacity models, instead raw spectrum resources are focused upon and is of the unit resource blocks. Where a resource block is a 1TS * 1BW segment. (The TS may be 5ms, and BW be 12.5kHz) ...

    #will also need to perform these analyses as a function of time, as the sectrum changes over time, so classic, throughput vs device plots are not entirely valid here, as there is an additional variable in play. 

    framemax = np.max(dataset['frame_no'])
    dev_arry = np.zeros(framemax)

    bwmin = 66;
    tsmin = 10; #these will not be required, as the windowing function already uses these basic values as the windows are observed.

    dev_bw = 66 #bandwidth for each device
    dev_ts = 11 #timescale for each device

    #use dataset, a subset here is not required

    dataset = dataset.sort_values(['frequency'], ascending=True)
    dataset['start'] = dataset['frame_no'].subtract(dataset['timescale']-1)

    #print('Printing frame 1')
    #print(dataset.loc[dataset['start'] == 1])

    ds_grp = dataset.groupby(dataset['start'])
    #ds_grp = ds_grp['frequency'].apply(lambda x: x.sort_values(ascending=False))

    #for c in ds_grp.groups: 
    #    print(c)
    #print(ds_grp.get_group(1))

    cur_frame = 1
    spect_df = pd.DataFrame()
    for key, group in ds_grp:
        if key < 13:
            print(key)
            #print(group)
            spect_df = spect_df.append(group)
            spect_df = spect_df[spect_df.frame_no > key] #this is pretty inefficient
            print(spect_df)
        #key is the identifier for the group - this is the start number
        #group becomes the entires within the group, ordered with ascending frequency
        #diff = 
        #cur_frame = key



if __name__ == "__main__":
    sys.exit(int(main() or 0))