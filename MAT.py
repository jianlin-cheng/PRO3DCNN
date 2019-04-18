import pickle as pick
#from pylab import *
from prody import *
# import Bio.PDB as pdb
import os
from numpy import array
from numpy import pad
from datetime import datetime
import math
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize,linewidth=170)

global verbose
verbose = True

def vprint(item):
    global verbose
    if (verbose):
        print(item)

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pick.load(fo, encoding='bytes')
    return data

def pickle(data,filename):
    f = open(filename,"wb")
    pick.dump(data,f,protocol=2)
    f.close()


# ====================================================================================
# DistM
# ====================================================================================

# Return a list of directories: = [index,folderName,filename,y]
def getProteinDirectory(dirpath):
    directory = []
    
    with open(dirpath) as infile:
        for index,line in enumerate(infile):
            linedata = line.split()
            hierachy = linedata[3].split('.')
            domain = linedata[0]
            folderName = domain[2:4]
            fileName = domain+'.ent'
            y = hierachy[0]+hierachy[1]

            if (hierachy[0]=='h'): #Quit protein directory after a-g
               break

            directory.append([index,folderName,fileName,y])
            
    return directory

def getChain(pdb_file):
    backbone = parsePDB(pdb_file, subset='ca')
    coordinates = backbone.getCoords()
    numCA = coordinates.shape[0]
    
    chain = []
    for i in range(0, numCA):
        chain.append(coordinates[i , 0:3])
    return chain

def getChains(loadPath, savePath):
    data = {b'x':[],b'y':[],b'id':[]}
    directory = getProteinDirectory('dir.cla.scop.1.55.txt')

    global errorList,chainN,saveSize,yLabels,lastLabel,lastLabelInt

    errorList = []
    chainN = 0
    saveSize = 10000
    def batchNStr():
        global errorList,chainN,saveSize
        return str(int(chainN/saveSize)-1)

    yLabels = {}
    lastLabel = ''
    lastLabelInt = 0
    def manageLabel():
        global yLabels,lastLabel,lastLabelInt
        if(lastLabel != chainLabel):
            lastLabelInt+=1
            lastLabel = chainLabel
            yLabels[lastLabelInt] = lastLabel


    for entry in directory:
        
        pdbPath = loadPath+'/'+entry[1]+'/'+entry[2]
        print(pdbPath)
        if os.path.exists(pdbPath):
            try:
                chain = getChain(pdbPath)
                data[b'x'].append(chain)
                data[b'id'].append(chainN)
                chainLabel = entry[3]
                manageLabel()
                data[b'y'].append(lastLabelInt)
                chainN += 1
            except AttributeError as error:
                errorList.append((entry,"noneType"))
        else:
            errorList.append((entry,"noEntFileFound"))

        if chainN%saveSize==0:
            pickle(data,savePath+'/chain'+batchNStr())
            data = {b'x':[],b'y':[],b'id':[]}


    if chainN%saveSize!=0:
        pickle(data,savePath+'/chain'+batchNStr())

    pickle(errorList,savePath+'/errorList')
    pickle(yLabels,savePath+'/yLabels')

def getDistM(chain):
    l = len(chain)
    distM = np.zeros([l, l])

    triu_indices = np.tril_indices(l,1)
    Is,Js = triu_indices[0],triu_indices[1]

    for k in range(len(Is)):
        i,j = Is[k],Js[k]
        distance = np.linalg.norm(chain[i]-chain[j])
        distM[i][j] = distance
        distM[j][i] = distance

    return distM

def sparseChain(chain,skip):
    nchain = [item for k, item in enumerate(chain) if k % skip != 0]
    return nchain

def getDistMs(loadPath,savePath,sparse):
    global matN,saveSize
    saveData = {b'x':[],b'y':[],b'id':[]}
    matN = 0
    saveSize = 1000

    def batchNStr():
        global matN,saveSize
        return str(int((matN+1)/saveSize)-1)

    ranges = [(0,5),(5,10),(10,15),(15,20),(20,24)]
    for i in ranges[4]:#range(0,23+1):
        database = unpickle(loadPath+"/chain"+str(i))
        chains = database[b'x']
        y = database[b'y']
        ids = database[b'id']

        for j,chain in enumerate(chains):
            matN = i*10000+j
            if(sparse):
                chain = sparseChain(chain,2)
            print(str(i)+" "+str(j)+" id: "+str(ids[j])+' matN:'+str(matN))
            distMatrix = getDistM(chain)
            saveData[b'x'].append(distMatrix)
            saveData[b'y'].append(y[j])
            saveData[b'id'].append(ids[j])
            
            if (matN+1)%saveSize==0:
                pickle(saveData,savePath+'/matBatch'+batchNStr())
                saveData = {b'x':[],b'y':[],b'id':[]}

    if (matN+1)%saveSize!=0:
        pickle(saveData,savePath+'/matBatch'+batchNStr())


def genChainAnalytics(loadPath,savePath):
    global matN,saveSize

    classN = {}
    chainLen = []

    for i in range(0,23+1):
        database = unpickle(loadPath+"/chain"+str(i))
        chains = database[b'x']
        y = database[b'y']
        ids = database[b'id']

        for j,chain in enumerate(chains):
            try:
                chainLen.append(len(chain))
                classN[y[j]]+=1
            except KeyError as error:
                classN[y[j]]=0


# ====================================================================================
# Main
# ====================================================================================
# getChains(loadPath='pdbstyle-1.55',savePath='chains')
# getChains(loadPath='pdbstyle-2.07/',savePath='/media/sec/chains')
# getDistMs(loadPath='/media/sec/chains',savePath='/media/sec/realMat',sparse=False)
# getSPDistMs(loadPath='/media/sec/chains',savePath='/media/sec/spDistMat')
