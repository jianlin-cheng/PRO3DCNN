import Graphs
import MAT
# chainData = MAT.unpickle('/Users/yhong/Desktop/SCOP_Simple/chains/chain_1.55_batch'+str(1))
# for pts in chainData[b'chain']:
#     if len(pts)>700:
#         break
pts = MAT.getChain('1ux8.pdb')

pts = MAT.sparseChain(pts,2)

distM = MAT.getDistM(pts)
Graphs.showDistMImage(distM)
Graphs.homPlot(pts,False)

# freqPlotChainLengths()