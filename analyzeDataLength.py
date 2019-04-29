import 

def genChainAnalytics(loadPath,savePath,sparse):

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


genChainAnalytics(loadPath='/media/sec/chains')
MAT.unpickle('/Users/yhong/Desktop/SCOP_Simple/chains/chain_1.55_batch'