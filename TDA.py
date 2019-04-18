import pickle as pick
import operator
import numpy as np
import math
import sys
import time
from sparseM import sparseM
from MAT import getDistM


def convertEdgesToVertices(Edges):
    v = []
    for edge in Edges:
        v.append(edge[0])
        v.append(edge[1])
    # print(v)
    return list(set(v))

def getEdgesFromCol(col, Edges):
    EC = []
    for i,c in enumerate(col):
        if(c!=0):
            EC.append(Edges[i])
    return EC


def getEdgesFromSparseCol(col, Edges):
    EC = []
    for i in col.keys():
        EC.append(Edges[i])
    return EC

def printMat(M):
    for r in range(M.shape[0]):
        print(M[r,:].tolist())

def sortedTupleFromList(s):
    list.sort(s)
    s = tuple(s)
    return s

def pickle(data,filename):
    f = open(filename,"wb")
    pick.dump(data,f)
    f.close()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def sparseMReduce(M,I):
    columnAtPivot = [None] * M.r
    columns = M.c
    M.zeroCols = []
    M.nonZeroCols = []

    for i in range(columns):
        
        pivotError = True
        while(True):
            currPivot = M.getPivot(i)

            if(currPivot == -1):
                M.zeroCols.append(i)
                break
            elif(columnAtPivot[currPivot] == None):
                M.nonZeroCols.append(i)
                columnAtPivot[currPivot] = i
                break
            else:

                j = columnAtPivot[currPivot]
                multFactor = -1* M.get(currPivot,i)/M.get(currPivot,j)
                M.addMultColtoCol(c1=i,m=multFactor,c2=j)
                I.addMultColtoCol(c1=i,m=multFactor,c2=j)

    return M,I


#Returns list of barcodes [V,(birth,death)]
def genHom(distM,useNewHomGen):
    # useNewHomGen = False
    info = []
    dArr = []
    maxSeqDist = 0
    
    l = distM.shape[0]
    
    maxLen = 0
    sumChainLen = 0
    chainSeg = []
    chainSegFactor = 1/5
    if(useNewHomGen):
        for i in range(l):
            for j in range(i,l):
                d = distM[i][j]
                if d!=0:
                    
                    if(j-i==1):
                        a = ((i,j),chainSegFactor*d)
                        chainSeg.append(a)
                        sumChainLen += d
                    else:
                        a = ((i,j),d)
                        dArr.append(a)
                    if(d>maxLen):
                        maxLen = d

    else:
        for i in range(l):
            for j in range(i,l):
                d = distM[i][j]
                if d!=0:
                    a = ((i,j),d)
                    dArr.append(a)
                    if(j-i==1):
                        chainSeg.append(a)
                        sumChainLen += d
                    if(d>maxLen):
                        maxLen = d

    info.append(('L:',l))
    # ===============================================================
    # Find dist array
    # ===============================================================

    dArr.sort(key=operator.itemgetter(1))
    chainSeg.sort(key=operator.itemgetter(1))
    t0 = time.time()

    # ===============================================================
    # Cut off :TODO Alternative is to do nothing if simplices is empty since not interesting simplices will form
    # ===============================================================
    # for subRad in range(3):

    #TODO: OPTION 1 Use FIXED DIAMETER DEPENDING ON SIZE
    # diameter = maxLen/6
    #TODO: OPTION 2 USE Average chain Len
    # diameter = sumChainLen/l*1.3
    #TODO: OPTION 3 USE Fixed
    diameter = 6.5


    # print(diameter)

    # print('maxLen: '+str(maxLenRad/(3))+'  seqDist: '+str(maxSeqDist))
    if(useNewHomGen):
        cutOffIndex = 0
        for i,p in enumerate(dArr):
            cutOffIndex=i
            if p[1] > diameter:#maxLenRad/(3-subRad):
                break

        tempdArr = dArr[:cutOffIndex]

        ebrth = {}
        Edges = []
        Dist = []
        for i,p in enumerate(tempdArr):
            ebrth[p[0]]=i
            Edges.append(p[0])
            Dist.append(p[1])
    else:
        cutOffIndex = 0
        for i,p in enumerate(dArr):
            cutOffIndex=i
            if p[1] > diameter:#maxLenRad/(3-subRad):
                break

        tempdArr = dArr[:cutOffIndex]
        for p in chainSeg:
            if p[1] > diameter:
                tempdArr.append(p)

        ebrth = {}
        Edges = []
        Dist = []
        for i,p in enumerate(tempdArr):
            ebrth[p[0]]=i
            Edges.append(p[0])
            Dist.append(p[1])
    # =================
    # Find Simplices
    # =================
    l = len(tempdArr)
    n = distM.shape[0]
    S = []
    SBirth = []
    RE = [[] for i in range(n)] # Related Edges
    for i,x in enumerate(tempdArr):
        e = x[0] #edge

        #Calcuate Simplexes Formed (For each Related Edge, add one at third point the edge it is connected to)
        D = [0]*n
        for r in RE[e[0]]:
            if (r[0]==e[0]) or (r[0]==e[1]):
                D[r[1]] += 1
                #AddSimplex
                if(D[r[1]] == 2):
                    j = r[1]
                    s = sortedTupleFromList([e[0],e[1],j])
                    S.append(s)
                    e1 = sortedTupleFromList([e[0],j])
                    e2 =  sortedTupleFromList([e[1],j])
                    SBirth.append(max([ebrth[e],ebrth[e1],ebrth[e2]]))

            else:
                D[r[0]] += 1
                if(D[r[0]] == 2):
                    j = r[0]
                    s = sortedTupleFromList([e[0],e[1],j])
                    S.append(s)
                    e1 = sortedTupleFromList([e[0],j])
                    e2 =  sortedTupleFromList([e[1],j])
                    SBirth.append(max([ebrth[e],ebrth[e1],ebrth[e2]]))
        
        for r in RE[e[1]]:
            if (r[0]==e[0]) or (r[0]==e[1]):
                D[r[1]] += 1
                if(D[r[1]] == 2):
                    j = r[1]
                    s = sortedTupleFromList([e[0],e[1],j])
                    S.append(s)
                    e1 = sortedTupleFromList([e[0],j])
                    e2 =  sortedTupleFromList([e[1],j])
                    SBirth.append(max([ebrth[e],ebrth[e1],ebrth[e2]]))
            else:
                D[r[0]] += 1
                if(D[r[0]] == 2):
                    j = r[0]
                    s = sortedTupleFromList([e[0],e[1],j])
                    S.append(s)
                    e1 = sortedTupleFromList([e[0],j])
                    e2 =  sortedTupleFromList([e[1],j])
                    SBirth.append(max([ebrth[e],ebrth[e1],ebrth[e2]]))


        RE[e[0]].append(e)
        RE[e[1]].append(e)

    # if len(S)>0:
    #     break
    if len(S)==0:
        return [],info
    dArr = tempdArr


    t1 = time.time()
    info.append(("Edgs:",len(dArr)))
    info.append(("Smplx:",round(t1-t0,5)))
    
    


    t0 = time.time()
    # =================
    # Construct D1
    # =================
    r = distM.shape[0]
    c = len(dArr)
    D1 = sparseM(r,c)
    # D1 = np.zeros((r,c))

    # I = np.identity(c)
    I = sparseM.I(c)
    for i,x in enumerate(dArr):
        e = x[0] #Edge Pair
        D1.set(e[0],i,-1)
        D1.set(e[1],i, 1)
    # TODO : Stack and implement mat until


    # Reduce D1
    # D1 = np.vstack((D1,I))
    R1,V1 = sparseMReduce(D1,I)

    t1 = time.time()
    info.append(('D1:',round(t1-t0,5)))
    # print('R1')
    # isReduced(R1)
    # printMat(R1)
    # print(R1)
    # print('V1')
    # print(V1)
    # printMat(V1)


    t0 = time.time()
    # =================
    # Construct D2
    # =================
    r = len(dArr)
    c = len(S)

    EDict = {}
    for i,x in enumerate(dArr):
        e = x[0]
        EDict[e] = i

    D2 = sparseM(r,c)
    I = sparseM.I(c)
    # print("Simplexes")
    # print(S)
    for i,s in enumerate(S):
        e1 = (s[0],s[1])
        e2 = (s[1],s[2])
        e3 = (s[0],s[2])
        D2.set(EDict[e1],i,1)
        D2.set(EDict[e2],i,1)
        D2.set(EDict[e3],i,-1)

    # Reduce D2
    
    R2,V2 = sparseMReduce(D2,I)

    t1 = time.time()
    info.append(('D2',round(t1-t0,5)))

    # print('R2')
    # printMat(R2)
    # isReduced(R2)
    # print(R2)
    # print('V2')
    # printMat(V2)
    # print(V2)
    Barcodes = []

    t0 = time.time()
    # ====================
    # Construct B from nonzero cols in R2 (Finite Barcode)
    # ====================

    B = []

    for i in R2.nonZeroCols:
        birth = R2.getPivot(i)
        death = SBirth[i]
        col = R2.cols[i]
        B.append(col)
        if(birth!=death):
            E = getEdgesFromSparseCol(col, Edges)
            V = convertEdgesToVertices(E)
            Barcodes.append([E,(Dist[birth],Dist[death])])
    # print('B')
    # print(np.transpose(np.array(B)))
    t1 = time.time()
    info.append(('B',round(t1-t0,5)))

    t0 = time.time()
    # ====================
    # Construct Z from V1 cols correspond to zero col in R1 (Infinite Barcode)
    # ====================

    Z = []

    for i in R1.zeroCols:
        col = R1.cols[i]
        Z.append(V1.cols[i])
    # print('Z')
    # print(np.transpose(np.array(Z)))
    t1 = time.time()
    info.append(('Z',round(t1-t0,5)))

    # ====================
    # Construct Main Obj
    # ====================

    t0 = time.time()

    Main = B.copy()
    BPivotList = []

    BPivotExists = [False] * R2.r
    for Bcol in B:
        BPivotExists[sparseM.getPivotOfCol(Bcol)] = True

    for i,Zcol in enumerate(Z):
        if(len(B) == len(Z)):
            break
        
        ZcolPivot = sparseM.getPivotOfCol(Zcol)

        if(not BPivotExists[ZcolPivot]):
            Main.append(Zcol)
            E = getEdgesFromSparseCol(Zcol, Edges)
            V = convertEdgesToVertices(E)
            birthIndex = sparseM.getPivotOfCol(Zcol)
            death = 'inf'
            Barcodes.append([E,(Dist[birthIndex],death)])

    # print("Main Object")
    # print(np.transpose(np.array(Main)))
    
    t1 = time.time()
    info.append(('Main',round(t1-t0,5)))
    info = [("nbcodes:",len(Barcodes))] + info
    print(info)
    return Barcodes,info
    # pickle(barcode,'barcode')

def genHoms(loadPath, savePath):

    saveData = {b'x':[],b'y':[],b'id':[]}
    c = 0
    for i in range(4):
        data = unpickle(loadPath+'/matBatch'+str(i))
        xs = data[b'x']
        ys = data[b'y']
        # ids = data[b'id']
        print(len(xs))
        for j,m in enumerate(xs):
            print(str(i)+'-'+str(j))
            if(m.shape[0]<=150):
                barcodes = genHom(m)
                saveData[b'x'].append(barcodes)
                saveData[b'y'].append(ys[j])
                # saveData[b'id'].append(ids[j])
                c +=1
        pickle(saveData,savePath+'/barcodes'+str(i))
        saveData = {b'x':[],b'y':[],b'id':[]}


# ================================================
# Generate Barcode Image from points(birth, persistence)
# ================================================
def getBcodeImg(points,xr,yr,xn,yn):
    M = np.zeros([xn, yn])
    xl = (xr[1]-xr[0])/xn
    yl = (yr[1]-yr[0])/yn

    for p in points:
        x = math.floor((p[0]-xr[0])/xl)
        y = math.floor((p[1]-yr[0])/yl)

        if(x<xn and y<yn):
            scale = (y/yn)*10
            M[x][y] += 1*scale

    return np.rot90(M)

def getBcodeImgs(loadPath, savePath):
    c = 0
    windowSize = 100
    # TODO: determine this range automatically
    xrng = (3.5,8.5)
    yrng = (0,2.5+1)
    infpersistence = 3.5-.1

    for i in range(4):
        saveData = {b'x':[],b'y':[],b'id':[]}
        data = unpickle(loadPath+'/barcodes'+str(i))
        X = data[b'x']
        Y = data[b'y']
        ids = data[b'id']
        for barcodes in X:
            print(c)
            c+=1
            plotPoints = []
            for barcode in barcodes:
                if( barcode[1][0]!=barcode[1][1]):
                    if(barcode[1][1] != 'inf'):
                        persistence = barcode[1][1] - barcode[1][0]
                        plotPoints.append((barcode[1][0],persistence))
                    else:
                        k = 1
                        plotPoints.append((barcode[1][0],infpersistence))
            M = getBcodeImg(plotPoints,xrng,yrng,windowSize,windowSize)
            saveData[b'x'].append(M)
        saveData[b'y'] = Y
        saveData[b'id'] = ids
        pickle(saveData,savePath+'/barcodeImage'+str(i))
        saveData = {b'x':[],b'y':[],b'id':[]}


# genHoms(loadPath='/Users/yhong/Desktop/SCOP_Simple/Ms',savePath='/Users/yhong/Desktop/SCOP_Simple/bcodes')
# getBcodeImgs(loadPath='/media/sec/bcodes',savePath='/media/sec/bcodeImgs')

# loadPath='/Users/yhong/Desktop/SCOP_Simple/Ms'
# c = 0
# for i in range(4):
#     data = unpickle(loadPath+'/matBatch'+str(i))
#     xs = data[b'x']
#     ys = data[b'y']
#     # ids = data[b'id']
#     print(len(xs))
#     for j,m in enumerate(xs):
#         print(str(i)+'-'+str(j))
#         if(m.shape[0]<=300):
#             barcodes1 = genHom(m)