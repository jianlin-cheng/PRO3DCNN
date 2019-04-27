import MAT
import TDA
import GRAPH
from sparseM import sparseM
from numpy import array
import numpy as np

# global N
# global chainData
#
# useNewMethod = False
# n=1
# N=-1
#
# batchSize = 0
# print(str(n//6000)+'-'+str(n%6000))
# if (N != n//6000):
#     N = n//6000
#     chainData = MAT.unpickle('/Users/yhong/Desktop/SCOP_Simple/chains/chain_1.55_batch'+str(N))[b'chain']
#
# chain = chainData[n%6000]
# GRAPH.homPlot(chain, useNewMethod)
# distM = MAT.getDistM(chain)

# GRAPH.freqPlotChainClasses('chains/chain_1.55_batch')




# Experiment 3
chain = MAT.getChain('1ux8.pdb')
distM = MAT.getDistM(chain)
barcode,info = TDA.genHom(distM, False)
image = TDA.getBcodeImgOne(barcode)
GRAPH.showDistMImage(image)
# # dist = 3.78
#
# dist = 5.4
# edges = TDA.getEdgesLengthLessThan(distM,dist)
# print(edges)
# GRAPH.plotChainWithEdge(chain,edges,dist)
# GRAPH.homPlot(chain,False)


#Experiment 4 generate graph for Jan
# chain = MAT.unpickle('chains/chain_1.55_batch0')[b'x'][1]
# GRAPH.homPlot(chain,useNewMethod=True)
# GRAPH.plotChainWithAllHom(chain,useNewMethod=True)

#Experiment 5 gendistmatrix image
# chain = MAT.getChain('1ux8.pdb')
# distM = MAT.getDistM(chain)
# GRAPH.showDistMImage(distM)

#Experiment 6 get the chain frequencies
# GRAPH.freqPlotChainLengths('chains/chain_1.55_batch')


#Experiment 7
# for i in range(5):
#     data = MAT.unpickle('sparseMs/barcodeimages'+str(i))[b'x']
#     print(len(data))

    # TDA.genHoms('sparseMs/sparseDistBatch','sparseMs/barcodes',5)
    # TDA.getBcodeImgs('sparseMs/barcodes','sparseMs/barcodeimages',5)


#Experiment 8: Reduce a matrix
# M = sparseM(4,6)
# I = sparseM.I(6)
#
# Mnp = [[-1, 0,-1,-1, 0, 0],
#        [ 0,-1, 1, 0,-1, 0],
#        [ 1, 1, 0, 0, 0,-1],
#        [ 0, 0, 0, 1, 1, 1]]
#
# Mnp = array(Mnp)
# M.setWithNParray(Mnp)
#
# M,I = TDA.sparseMReduce(M,I)
# print(M.nparray())
# print(I.nparray())


#Experiment 9: Graph problem chain with inter distance longer than 60
# chain = TDA.unpickle('chain')
# # GRAPH.plotChain(chain)

#Experiment 10: print out homology process for a chain in 2D
chain = [[-1.56752,0.806667,0],[-1.47077,0.441724,0],[-1.35437,0.00714125,0],[-1.02374,-0.391121,0],[-0.46443,-0.579549,0],[-0.225042,0.964776,0],[0.117617,-0.73655,0],[0.743939,-0.693085,0],[0.864936,1.06756,0],[1.44047,-0.95066,0],[1.54389,0.992477,0],[1.94542,0.96517,0],[2.04827,0.806754,0],[2.29066,0.199273,0],[2.42265,-0.523491,0],[2.97549,-0.416042,0],[3.14736,0.006578,0]]
# chain = [[-1.56752,0.806667,0],[-1.47077,0.441724,0],[-1.35437,0.00714125,0],[-1.02374,-0.391121,0],[-0.46443,-0.579549,0],[-0.225042,0.964776,0],[0.117617,-0.73655,0],[0.743939,-0.693085,0],[0.864936,1.06756,0],[1.44047,-0.95066,0],[1.54389,0.992477,0]]
print(len(chain))
chain = array(chain)

distM = MAT.getDistM(chain)
# print(np.around(distM,2))
# edges = TDA.getEdgesLengthLessThan(distM,0)
# GRAPH.plotChainWithEdge(chain,edges,False)
# GRAPH.homPlot(chain,False)