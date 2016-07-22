import tkinter as tk
from tkinter import filedialog
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import cProfile, pstats, io

__author__ = 'Pierce Rixon'
select = 1

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
    filename = filedialog.askopenfilename() #('Window_dump.csv')

    analyse(filename,select)
    #legacy_sns(filename)

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

        ### PLOTTING THINGS HERE ###

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
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        #ax.set_xlim(0,pxls)
        #ax.set_ylim(0,pxls)

        ax.set_title('Window Count')
        ax.set_xlabel('Bandwidth in Bins (190Hz/bin)')
        ax.set_ylabel('Timescale (5.3ms/unit)')

        ax2.set_title('Whitespace Density')
        ax2.set_xlabel('Bandwidth in Bins (190Hz/bin)')
        ax2.set_ylabel('Timescale (5.3ms/unit)')
    
        plt.show()

        #1.b
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
        ax2.set_yscale('log')
        ax2.set_xscale('log')

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
        ax2a.set_yscale('log')
        ax2a.set_xscale('log')

        axa.set_title('Bandwidth Distribution')
        axa.set_xlabel('Bandwidth in Bins (190Hz/bin)')
        axa.set_ylabel('Number of Observations')

        ax2a.set_title('Timescale Distribution')
        ax2a.set_xlabel('Timescale (5.3ms/unit)')
        ax2a.set_ylabel('Number of Observations')

        plt.show()

    #elif(test == 8):
    #BW vs WS and TS vs WS analysis - Test 1 is the complementary analysis of this test
    #we want to plot both unique whitespace incurred as well as total whitespace as the x axis decreases

        

    elif(test == 2):
    #2. WS vs Bins (real frequency - may need to preserve band starting frequency ....) 
    #Series 1 - WS per bin due to partitioning algorithm
    #Series 2 - WS per bin based on the unpartitioned spectrum
        
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

        if True:
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
        relwsS = np.sort(relws)

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



if __name__ == "__main__":
    sys.exit(int(main() or 0))