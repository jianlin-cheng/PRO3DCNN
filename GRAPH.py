import numpy as np

import sys


import MAT 
import TDA
import ytl
import ast

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pick
import operator
from matplotlib.backend_bases import key_press_handler
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


np.set_printoptions(threshold=sys.maxsize,linewidth=170)

global barcodeIndex
global barcodes
global ax
global pts
global fig

def press(event):
    global ax
    global barcodes
    global barcodeIndex
    global pts
    global fig
    print('press', event.key)

    draw = False
    
    if event.key == 'right':
        if(barcodeIndex < len(barcodes)):
            barcodeIndex +=1 
        draw = True
    elif event.key == 'left':
        if(barcodeIndex > 0):
            barcodeIndex -=1 
        draw = True
    elif event.key == 'down':
        if(barcodeIndex-5 > 0):
            barcodeIndex -=5 
        draw = True
    elif event.key == 'up':
        if(barcodeIndex+5 < len(barcodes)):
            barcodeIndex +=5 
        draw = True


    if (draw):
        b = barcodes[barcodeIndex]
        print(b)
        ax.cla()
        # fig.cla()
        b=barcodes[barcodeIndex]

        # for b in barcodes:
        xPoints = [point[0] for point in pts]
        yPoints = [point[1] for point in pts]
        zPoints = [point[2] for point in pts]
        # ax.scatter(xPoints, yPoints, zPoints, 'b')
        ax.plot(xPoints,yPoints,zPoints,'b')
        
        print(b)
        points = np.array([p.tolist() for p in pts])
        edges =  np.array([[e[0],e[1]] for e in b[0]])
        print(points)
        print(edges)
        lc = Line3DCollection(points[edges],colors='r')
        plt.gca().add_collection(lc)
        plt.xlim(points[:,0].min(), points[:,0].max())
        plt.ylim(points[:,1].min(), points[:,1].max())
        # plt.zlim(points[:,2].min(), points[:,2].max())
        



        birth = round(b[1][0],4)
        if (b[1][1] != 'inf'):
            death = round(b[1][1],4)
        else:
            death = 'inf'
        per = round(b[2],4)
        plt.plot(points[:,0], points[:,1],points[:,2],'b',label="interval: "+str((birth,death))+" persist: "+str(per)+" "+str(barcodeIndex)+'/'+str(len(barcodes)-1))
        ax.legend()

        plt.show()
        plt.draw()

def freqPlot(pts,num_bins,title):
    # num_bins = 10
    n, bins, patches = plt.hist(pts, num_bins, facecolor='blue', alpha=0.5)
    plt.title(title)
    plt.show()

def freqPlotChainLengths(loadPath):
    c=0
    pts = []

    for i in range(4+1):
        print(i)
        chainData = MAT.unpickle(loadPath+str(i))
        TDA.pickle(chainData,loadPath+str(i))
        for j,chain in enumerate(chainData[b'x']):
            pts.append(len(chain))

    freqPlot(pts,30,'Chain Lengths')

def freqPlotChainClasses(loadPath):
    pts = []

    classCount = {}

    for i in range(4+1):
        print(i)
        chainData = MAT.unpickle(loadPath+str(i))
        for ylbl in chainData[b'y']:
            try:
                classCount[ylbl] += 1
            except KeyError as error:
                classCount[ylbl] = 0
            pts.append(ylbl)

    classCounts = []
    for k in classCount:
        classCounts.append(classCount[k])
    print(np.median(classCounts))

    numClasses = ylbl

    freqPlot(pts,numClasses,'Proteins Per Fold')
    freqPlot(classCounts, 1000, '# per Fold')

def showDistMImageNeg(distM):
    from PIL import Image
    import PIL.ImageOps
    m = np.amax(distM)
    imageM = distM*255/m
    imageM = imageM*-1
    imageM = imageM+255
    imageM = imageM.astype(np.uint8)
    im = Image.fromarray(imageM)
    im.show()

def showDistMImage(distM):
    from PIL import Image
    import PIL.ImageOps
    m = np.amax(distM)
    imageM = distM*255/m
    imageM = imageM*-1
    imageM = imageM+255
    imageM = imageM.astype(np.uint8)
    im = Image.fromarray(imageM)
    im.show()

def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def plotChainWithAllHom(chain,useNewMethod):
    distM = MAT.getDistM(chain)
    barcodes, info = TDA.genHom(distM,useNewMethod)
    edges = []

    for b in barcodes:
        edges += b[0]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for b in barcodes:
    xPoints = [point[0] for point in chain]
    yPoints = [point[1] for point in chain]
    zPoints = [point[2] for point in chain]

    ax.scatter(xPoints, yPoints, zPoints, 'r')

    points = np.array([p.tolist() for p in chain])
    lc = Line3DCollection(points[edges], colors='r')
    # Edges
    plt.gca().add_collection(lc)

    plt.draw()
    plt.show()
    plt.pause(1000000)

def plotChainWithEdge(chain,edges,dist):
    pts = chain


    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for b in barcodes:
    xPoints = [point[0] for point in pts]
    yPoints = [point[1] for point in pts]
    zPoints = [point[2] for point in pts]

    #Chain
    # ax.plot(xPoints, yPoints, zPoints, 'b')

    #Balls
    # ax.scatter(xPoints, yPoints, zPoints, 'r', s=350, alpha=0.4)

    #Points
    ax.scatter(xPoints, yPoints, zPoints, 'r')

    radius = [dist]*len(pts)
    for (xi, yi, zi, ri) in zip(xPoints, yPoints,zPoints, radius):
        (xs, ys, zs) = drawSphere(xi, yi, zi, ri)
        # ax.plot_surface(xs, ys, zs, color="b", alpha=.1)


    points = np.array([p.tolist() for p in pts])
    lc = Line3DCollection(points[edges],colors='r')
    #Edges
    plt.gca().add_collection(lc)


    plt.draw()
    plt.show()
    plt.pause(1000000)

def plotChain(chain):
    pts = chain

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for b in barcodes:
    xPoints = [point[0] for point in pts]
    yPoints = [point[1] for point in pts]
    zPoints = [point[2] for point in pts]

    ax.plot(xPoints, yPoints, zPoints, 'r')


    plt.draw()
    plt.show()
    plt.pause(1000000)

def homPlot(chain,useNewMethod):
    global barcodeIndex
    global barcodes
    global ax
    global pts
    global fig
    pts = chain
    distM = MAT.getDistM(pts)


    barcodes,info = TDA.genHom(distM,useNewMethod)

    print("nbarcodes"+str(len(barcodes)))

    for b in barcodes:
        interval = b[1]
        #Sort by length
        if (interval[1]=='inf'):
            b.append(10000000000-interval[0])
        else:   
            b.append(interval[1]-interval[0])
        # b.append(interval[0])
        # list.sort(b[0])
        b[0].append(b[0][0])

    barcodes.sort(key=operator.itemgetter(2))
    barcodeIndex = 0
    b=barcodes[barcodeIndex]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.mpl_connect('key_press_event', press)

    # for b in barcodes:
    xPoints = [point[0] for point in pts]
    yPoints = [point[1] for point in pts]
    zPoints = [point[2] for point in pts]

    ax.plot(xPoints,yPoints,zPoints,'b')
    

    points = np.array([p.tolist() for p in pts])
    edges =  np.array([[e[0],e[1]] for e in b[0]])
    lc = Line3DCollection(points[edges],colors='r')
    plt.gca().add_collection(lc)
    plt.xlim(points[:,0].min(), points[:,0].max())
    plt.ylim(points[:,1].min(), points[:,1].max())
    # plt.zlim(points[:,2].min(), points[:,2].max())
    



    birth = round(b[1][0],4)
    if (b[1][1] != 'inf'):
        death = round(b[1][1],4)
    else:
        death = 'inf'
    per = round(b[2],4)
    plt.plot(points[:,0], points[:,1],points[:,2],'b',label="interval: "+str((birth,death))+" persist: "+str(per)+"         "+str(barcodeIndex)+'/'+str(len(barcodes)-1))
    ax.legend()

    plt.draw()
    plt.show()
    plt.pause(1000000)
    return info


def homPlotPts(points, useNewMethod):
    global barcodeIndex
    global barcodes
    global ax
    global pts
    global fig
    pts = points
    distM = MAT.getDistM(pts)

    barcodes, info = TDA.genHom(distM, useNewMethod)

    print("nbarcodes" + str(len(barcodes)))

    for b in barcodes:
        interval = b[1]
        # Sort by length
        if (interval[1] == 'inf'):
            b.append(10000000000 - interval[0])
        else:
            b.append(interval[1] - interval[0])
        # b.append(interval[0])
        # list.sort(b[0])
        b[0].append(b[0][0])

    barcodes.sort(key=operator.itemgetter(2))
    barcodeIndex = 0
    b = barcodes[barcodeIndex]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.mpl_connect('key_press_event', press)

    # for b in barcodes:
    xPoints = [point[0] for point in pts]
    yPoints = [point[1] for point in pts]
    zPoints = [point[2] for point in pts]
    ax.scatter(xPoints, yPoints, zPoints, 'b')
    # ax.plot(xPoints, yPoints, zPoints, 'b')

    points = np.array([p.tolist() for p in pts])
    edges = np.array([[e[0], e[1]] for e in b[0]])
    lc = Line3DCollection(points[edges], colors='r')
    plt.gca().add_collection(lc)
    plt.xlim(points[:, 0].min(), points[:, 0].max())
    plt.ylim(points[:, 1].min(), points[:, 1].max())
    # plt.zlim(points[:,2].min(), points[:,2].max())

    birth = round(b[1][0], 4)
    if (b[1][1] != 'inf'):
        death = round(b[1][1], 4)
    else:
        death = 'inf'
    per = round(b[2], 4)
    # plt.plot(points[:, 0], points[:, 1], points[:, 2], 'b',
    #          label="interval: " + str((birth, death)) + " persist: " + str(per) + "         " + str(
    #              barcodeIndex) + '/' + str(len(barcodes) - 1))
    ax.legend()

    plt.draw()
    plt.show()
    plt.pause(1000000)
    return info

def inspectProblemChains():
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
            if (i[0]!='L:' and i[0]!='Edgs:'):
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
