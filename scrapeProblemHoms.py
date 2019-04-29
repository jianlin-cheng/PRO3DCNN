import ytl
import ast
import MAT 
import TDA
import GRAPH
import TDAOld


def genHomAtMat(n):
    global N
    global matData
    batchSize = 0
    num = n//1000
    subn = n-1000*num
    print(str(num)+'-'+str(subn))
    if (N != num):
        N = num
        matData = MAT.unpickle('/Users/yhong/Desktop/SCOP_Simple/sparseMs/sparseDistBatch'+str(N))

    distM = matData[b'x'][subn]
    barcodes,info = TDA.genHom(distM,False)
    return info


def graphHomAtChain(n,useNewMethod):
    global N
    global chainData
    batchSize = 0
    print(str(n//6000)+'-'+str(n%6000))
    if (N != n//6000):
        N = n//6000
        chainData = MAT.unpickle('/Users/yhong/Desktop/SCOP_Simple/chains/chain_1.55_batch'+str(N))[b'chain']

    chain = chainData[n%6000]
    GRAPH.homPlot(chain, useNewMethod)
    distM = MAT.getDistM(chain)

    return info

def gen3HomAtChain(n):
    global N
    global chainData
    batchSize = 0
    print(str(n//6000)+'-'+str(n%6000))
    if (N != n//6000):
        N = n//6000
        chainData = MAT.unpickle('/Users/yhong/Desktop/SCOP_Simple/chains/chain_1.55_batch'+str(N))[b'chain']

    chain = chainData[n%6000]
    distM = MAT.getDistM(chain)
    print('Chain Information')
    newChain = [c.tolist() for c in chain]
    print(newChain)
    print('old Method -------')
    barcodes,info = TDAOld.genHom(distM)
    print(barcodes)
    print('My Method -------')
    barcodes,info = TDA.genHom(distM,False)
    print(barcodes)
    print('Suggested Variant Method-------')
    barcodes,info = TDA.genHom(distM,True)
    print(barcodes)
    return info



def optimizeProblemExperiment():
    rawO = ytl.writeFileToList('homgenoutput')
    def process2(rawinfo):
        info = ast.literal_eval(rawinfo)
        return info
    def process1(numberStr):
        numI = numberStr.split('-')
        return int(numI[0])*1000+int(numI[1])


    O = [(process1(rawO[2*i]),process2(rawO[2*i+1])) for i in range(len(rawO)//2)]

    totalT = 0
    problemT = 0
    problemO = []

    def sumTimeTaken(info):
        t = 0
        for i in info:
            if (i[0]!='L:' and i[0]!='Edgs:' and i[0]!='nbcodes:'):
                t+=i[1]
        return t

    for o in O:
        t = sumTimeTaken(o[1])
        totalT += t
        if t > 60:
            problemO.append(o)
            problemT += t


    global N
    global matData
    global chainData
    N = -1
    problemTFix = 0
    for o in problemO:
        problemTFix += sumTimeTaken(genHomAtChain(o[0]))



    print(totalT)
    print(problemT)
    print(problemTFix)
    print(len(problemO))

# ========================
# main
# ========================
global N
global matData
global chainData
N =-1
# for n in range(4):
#     print('================================================================')
#     print('Generating homologies of protein '+str(n)+' with variant methods')
#     gen3HomAtChain(n)



graphHomAtChain(1,False)