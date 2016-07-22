import tkinter as tk
from tkinter import filedialog
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
import sys

Nrange = 128
rIdx = 1024
Ntime = 300

debug = False
debug2 = False

class Window:
    def __init__(self, ts, f, bw, ws, fn, val):
        self.ts = ts
        self.f = f
        self.bw = bw
        self.ws = ws
        self.fn = fn
        self.val = val
     
class BlockIndex: 
    def __init__(self, tID, rID):

        self.timeID = tID
        self.rangeID = rID
        self.idxArry  = []

    def add_win(self, win, idx):
        if ((win.f >= self.rangeID*Nrange & win.f < (self.rangeID+1)*Nrange) | (win.f <= self.rangeID*Nrange & win.f + win.bw >= self.rangeID*Nrange)) & \
            ((win.fn >= self.timeID * Ntime & win.fn < (self.timeID+1)*Ntime) | (win.fn > (self.timeID+1)*Ntime & win.fn - win.ts <= (self.timeID+1)*Ntime)):

            self.idxArry.append(idx)
        else: 
            print('This window should not be here. Block ID: {},{} ({},{}). Window start: {} end: {} at frame: {} ts: {}'.format(self.timeID, self.rangeID, self.timeID*Ntime, self.rangeID*Nrange, win.f, win.f + win.bw, win.fn, win.ts))

    def size(self):
        return len(self.idxArry)

    def entries(self):
        return self.idxArry

class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v

    def connect(self):
        temp = [u,v]
        return temp

    def print(self):
        print('{} - {}'.format(u,v))

def main():
    #ws='whitespace'
    #ts='timescale'
    #f='frequency'
    #bw='bandwidth'
    #fn='frame_no'

    matplotlib.style.use('ggplot')

    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename() #('Window_dump.csv')

    #graph(filename)
    legacy(filename)

def graph ( file ):

    VertexList = [] #V - list of vertices with the index of the vertex (window) list equal to the index of the window
    EdgeList = [] #E -

    BlockList = [] #B - psuedo 2d array of blocks. ... this actually could work well with a GPU architecture ................

    wins = pd.read_csv(file, header=0) #, chunksize = chnk)
    print(wins.columns)
     
    ######row[1]     row[2]    row[3]     row[4]     row[5]
    #cols:timescale,frequency,bandwidth,whitespace,frame_no
    cols = wins.columns

    t_thresh = 10
    b_thresh = 66

    candidate = 0
    total = 0
    counter = 0

    blockid = 0

    Nrows = 0
    Ncols = 0

    for row in wins.itertuples():
        total+=1
        winval = (row[1]- t_thresh)/t_thresh * (row[3] / b_thresh)
        if winval > 1 :
            
            temp = Window(int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5]),winval);

            VertexList.append(temp)

            #generate blocks (if necessary)
            if int(row[5]/Ntime) + 1 > int(blockid/rIdx):
                for x in range(0,rIdx):
                    BlockList.append(BlockIndex(int(blockid/rIdx),blockid%rIdx))
                    blockid += 1
                print('Blockid = {}'.format(blockid))

            #need to generate this list for ALL blocks covered, not just the first fit.
            #BlockList[int(row[5]/Ntime)*rIdx + int(row[2]/Nrange)].add_win(temp,candidate)

            for t in range(int((row[5] - row[1])/Ntime),int(row[5]/Ntime)+1): 
                for b in range(int(row[2]/Nrange),int((row[2] + row[3])/Nrange)+1):
                    
                    BlockList[t*rIdx + b].add_win(temp,candidate)

                    if debug: print('added candidate: {},{} to: {},{}'.format(row[5],row[2],t*rIdx,b))
            #generate edges

            edgeGen(VertexList,EdgeList,BlockList,temp,candidate)

            candidate += 1
        if total%1000 == 0:
            print('Row: {}, with {} windows'.format(total,candidate))
            print('Size of V: {}. Size of E: {}'.format(len(VertexList),len(EdgeList)))

    print('Total windows: {} Candiate windows: {}'.format(total,candidate))
    print('Final Size of V: {}. Size of E: {}'.format(len(VertexList),len(EdgeList)))


def edgeGen(V,E,B,win,id):
        #Nrows = row[3]/Nrange
        #Ncols = row[1]/Ntime
        #for t in range(row[5]/Ntime - Ncols,row[5]/Ntime):
        #    for b in range(row[2]/Nrange - Nrows,row[2]/Ntime):
        edgeCheck = []

        Nrows = int(win.ts/Ntime)
        Ncols = int(win.bw/Nrange)
        if debug2: print('Active block: {},{} with time {} and range {} - window details: f{}, bw{}, fn{}, ts{}'.format(int(win.fn/Ntime - Nrows),int(win.f/Nrange),Nrows,Ncols,win.f, win.bw, win.fn, win.ts))
        for t in range(int((win.fn - win.ts)/Ntime),int(win.fn/Ntime)+1): 
            for b in range(int(win.f/Nrange),int((win.f + win.bw)/Nrange)+1):
                   
                if debug2: print('Checking: {},{}'.format(t,b))
                edgeCheck += B[t*rIdx+b].entries()
                
        #print(edgeCheck)
        reducedEdge = np.unique(edgeCheck)
        if debug: print('id {}, possible conflict set: {}'.format(id, reducedEdge))

        for e in reducedEdge:
            unique = True
            test = V[e]
            if debug2: print('Current edge check id: {}'.format(e))
            if ((e != id) and ((test.f >= win.f and test.f < win.f+win.bw) or (test.f <= win.f and test.f + test.bw - 1 >= win.f)) and ((
                test.fn >= win.fn - win.ts + 1 and test.fn < win.fn) or (test.fn > win.fn and test.fn - test.ts < win.fn))):

                edge = Edge(id,e)
                E.append(edge)
                edge.print

                if debug: print('window id {}, ({},{} bw {} ts {}) overlapped by: id {}, ({},{} bw {} ts {})'.format(id, V[id].f, V[id].fn, V[id].bw, V[id].ts, e, V[e].f, V[e].fn, V[e].bw, V[e].ts,))
   
def ts_vs_ws ( file ):

    wins = pd.read_csv(file, header=0) #, chunksize = chnk)
    print(wins.columns)

    #cols:timescale,frequency,bandwidth,whitespace,frame_no
    cols = wins.columns
    
    ts = max(wins[cols[0]])
    fn = max(wins[cols[4]])

    ts_arr = np.arange(1,ts+1)
    ws_arr = np.zeros_like(ts_arr)

    totalws = 0

    #row:index,timescale,frequency,bandwidth,whitespace,frame_no
    for row in wins.itertuples():
        ws_arr[row[1]-1] += row[4]
        totalws += row[4]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.semilogy(ts_arr,ws_arr, label='Whitespace vs TS')

    ax.set_yscale('symlog') #super important to plot '0' values
    plt.text(0.2,0.8,"Total ws: " + "{:,}".format(totalws), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.legend()
    plt.show()



def legacy (filename):

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
    plt.show()

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