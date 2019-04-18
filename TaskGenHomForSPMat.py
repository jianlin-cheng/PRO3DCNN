import os
import TDA
import ytl


script_dir = os.path.abspath(__file__)
i = script_dir.rfind('/')
script_dir = script_dir[0:i]

for i in range(29+1):
    data = ytl.unpickle(script_dir+'/sparseMs/sparseDistBatch'+str(i))
    saveData = {b'x':[],b'y':[]}
    for j,M in enumerate(data[b'x']):
        print(str(i)+'-'+str(j))
        saveData[b'x'].append(TDA.genHom(M))
    saveData[b'y'] = data[b'y']
    ytl.pickle(saveData,script_dir+'/sparseMs/barcodes'+str(i))
    saveData = {b'x':[],b'y':[]}