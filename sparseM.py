import numpy as np

class sparseM:

    def __init__(self, r,c):
        self.r = r
        self.c = c

        self.cols = [{} for i in range(c)]

    @staticmethod
    def I(n):
        I = sparseM(n,n)
        for i in range(n):
            I.set(i,i,1)
        return I

    @staticmethod
    def getPivotOfCol(col):
        if(len(col)==0):
            return -1
        else:
            nonzeroRows = col.keys()
            return max(nonzeroRows)

    def set(self,r,c,val):
        if(val == 0):
            try:
                del self.cols[c][r]
            except KeyError as error:
                t='donothing'
        else:
            self.cols[c][r] = val

    def get(self,r,c):
        try:
            val = self.cols[c][r]
            return val
        except KeyError as error:
            return 0   

    def getPivot(self,c):
        col = self.cols[c]
        if(len(col)==0):
            return -1
        else:
            nonzeroRows = col.keys()
            return max(nonzeroRows)

    def setWithNParray(self,nparray):
        for i in range(nparray.shape[0]):
            for j in range(nparray.shape[1]):
                self.set(i,j,nparray[i][j])

    def nparray(self):
        M = np.zeros((self.r,self.c))
        for c,col in enumerate(self.cols):
            for r in col.keys():
                M[r][c]=col[r]
        return M

    def addMultColtoCol(self,c1,m,c2):
        col1 = self.cols[c1]
        col2 = self.cols[c2]
        for r in col2.keys():
            try:
                val = col1[r]+m*col2[r]
                if(val==0):
                    del col1[r]
                else:
                    col1[r]=val
            except KeyError as error: #col1's rth row is zero
                col1[r]=m*col2[r]

    def printinfo():
        print("r:"+self.r +" c:"+self.c)


test = sparseM(1,2)
# I = sparseM.I(10)
# for i in range(10):
#     print(I.getPivot(i))
# for i in range(10):
#     I.addMultColtoCol(0,-1,i)
# print(I.nparray())
# for i in range(10):
#     I.addMultColtoCol(0,-1,i)
# print(I.nparray())



