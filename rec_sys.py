from DecisionTree import DecisionTree
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

class Dataset:
    def __init__(self, folderPath, daSubset):
        self.folderPath = folderPath
        self.dataSubset = daSubset
        self._readData()

    def _readData(self):
        fileNames = ["u.user", "u.data", "u.item"]
        fsep = ["|", "\t", "|"]
        self.fCols = [
                    ['user id', 'age', 'gender', 'occupation', 'zip code'],
                    ['user id', 'item id', 'rating', 'watched on'],
                    ['item id', 'movie title', 'release date', 'video release date', 'IMDb URL', \
                        'unknown', 'Action', 'Adventure', 'Animation', "Children's",
                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        ]
        self.ret = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', \
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', \
            'Sci-Fi', 'Thriller', 'War', 'Western', 'tDiff', 'rating'
        ]

        st = time.time()
        self.fullData = {}
        self.trainData = {}
        self.testData = {}

        for fname, sep, col in zip(fileNames, fsep, self.fCols):
            self.fullData[fname] = pd.read_csv(self.folderPath + "/" + fname, sep=sep, names=col)
        
        for i in range(1, 8):
            if i <= 5:
                fTrName = "u" + str(i) + ".base" 
                fTsName = "u" + str(i) + ".test" 
            elif i == 6:
                fTrName = "ua.base"
                fTsName = "ua.test"
            elif i == 7:
                fTrName = "ub.base"
                fTsName = "ub.test"
            self.trainData[fTrName] = pd.read_csv(self.folderPath + "/" + fTrName , sep="\t", names=self.fCols[1] )
            self.testData[fTsName]  = pd.read_csv(self.folderPath + "/" + fTsName , sep="\t", names=self.fCols[1] )

            self.trainData[fTrName]["watched on"] = \
                pd.to_datetime( self.trainData[fTrName]["watched on"] , unit='s')
            self.testData[fTsName]["watched on"] = \
                pd.to_datetime( self.testData[fTsName]["watched on"] , unit='s')
        
        print("Read data in", str.format("{0:.3f}", time.time()-st) + "s" )

        self.fullData["u.item"]["release date"] = \
            pd.to_datetime( self.fullData["u.item"]["release date"] , format='%d-%b-%Y')
        self.fullData["u.data"]["watched on"] = \
            pd.to_datetime( self.fullData["u.data"]["watched on"] , unit='s')

    def getUserMovies(self, userID, mode):
        if mode == "train":
            df = self.trainData[self.dataSubset + ".base"]
        else:
            df = self.testData[self.dataSubset + ".test"]
        umv = df.loc[df["user id"] == userID]
        return umv[['item id', 'rating', 'watched on']]

    def getUserData(self, userID, mode):
        
        mw = self.fullData["u.item"]           
        toRet = self.getUserMovies(userID, mode).merge(  mw[self.fCols[2][-19:] + ['item id', 'release date'] ], how="inner"  )
        timeDiff = (toRet["release date"] - toRet["watched on"]).dt.days
        toRet["tDiff"]=timeDiff
        toRet["release date"] = (toRet["release date"] - pd.datetime(1970,1,1)).dt.days
        toRet = toRet.drop(columns=["watched on", "release date", "item id"])
        toRet.fillna(-100, inplace=True)
        return toRet[self.ret]

    def getGenreDict(self):
        ret = {}
        with open(self.folderPath + "/" "u.genre") as fh:
            for row in fh.readlines():
                l = row.strip().split("|")
                if len(l) == 2:
                    ret[int(l[1])] = l[0]
        return ret

def getEnums(arr):
    ret1 = (arr[:,0]*1 + arr[:,1]*5 +arr[:,2]*3 +arr[:,3]*7 +arr[:,4]*11 +arr[:,5]*13 +arr[:,6]*17 +arr[:,7]*19 +arr[:,8]*23 +arr[:,9]*29 +arr[:,10]*31 +arr[:,11]*37 +arr[:,12]*41 +arr[:,13]*43 +arr[:,14]*47 +arr[:,15]*53 +arr[:,16]*59 +arr[:,17]*67 +arr[:,18]*61).reshape(arr.shape[0], 1)
    ret2 = (arr[:,0]*1 + arr[:,1]*67 + arr[:,2]*61 + arr[:,3]*59 + arr[:,4]*53 + arr[:,5]*47 + arr[:,6]*43 + arr[:,7]*41 + arr[:,8]*37 + arr[:,9]*31 + arr[:,10]*29 + arr[:,11]*23 + arr[:,12]*19 + arr[:,13]*17 + arr[:,14]*13 + arr[:,15]*11 + arr[:,16]*7 + arr[:,17]*3 + arr[:,18]*5).reshape(arr.shape[0], 1)
    ret3 = (arr[:,0 ]*(1 )+arr[:,1 ]*(2**1.1 )+arr[:,2 ]*(2**1.2 )+arr[:,3 ]*(2**1.3 )+arr[:,4 ]*(2**1.4 )+arr[:,5 ]*(2**1.5 )+arr[:,6 ]*(2**1.6 )+arr[:,7 ]*(2**1.7 )+arr[:,8 ]*(2**1.8 )+arr[:,9 ]*(2**1.9 )+arr[:,10]*(2**2.0 )+arr[:,11]*(2**2.1 )+arr[:,12]*(2**2.2 )+arr[:,13]*(2**2.3 )+arr[:,14]*(2**2.4 )+arr[:,15]*(2**2.5 )+arr[:,16]*(2**2.6 )+arr[:,17]*(2**2.7 )+arr[:,18]*(2**2.8 )).reshape(arr.shape[0], 1)
    ret4 = (arr[:,0 ]*(1 )+arr[:,1 ]*(2**2.8 )+arr[:,2 ]*(2**2.7 )+arr[:,3 ]*(2**2.6 )+arr[:,4 ]*(2**2.5 )+arr[:,5 ]*(2**2.4 )+arr[:,6 ]*(2**2.3 )+arr[:,7 ]*(2**2.2 )+arr[:,8 ]*(2**2.1 )+arr[:,9 ]*(2**2.0 )+arr[:,10]*(2**1.9 )+arr[:,11]*(2**1.8 )+arr[:,12]*(2**1.7 )+arr[:,13]*(2**1.6 )+arr[:,14]*(2**1.5 )+arr[:,15]*(2**1.4 )+arr[:,16]*(2**1.3 )+arr[:,17]*(2**1.2 )+arr[:,18]*(2**1.1 )).reshape(arr.shape[0], 1)
    ret5 = (arr[:,0 ]*(2**0 )+arr[:,1 ]*(2**1 )+arr[:,2 ]*(2**2 )+arr[:,3 ]*(2**3 )+arr[:,4 ]*(2**4 )+arr[:,5 ]*(2**5 )+arr[:,6 ]*(2**6 )+arr[:,7 ]*(2**7 )+arr[:,8 ]*(2**8 )+arr[:,9 ]*(2**9 )+arr[:,10]*(2**10 )+arr[:,11]*(2**11 )+arr[:,12]*(2**12 )+arr[:,13]*(2**13 )+arr[:,14]*(2**14 )+arr[:,15]*(2**15 )+arr[:,16]*(2**16 )+arr[:,17]*(2**17 )+arr[:,18]*(2**18 )).reshape(arr.shape[0], 1)
    ret6 = (arr[:,0 ]*(2**0 )+arr[:,1 ]*(2**18 )+arr[:,2 ]*(2**17 )+arr[:,3 ]*(2**16 )+arr[:,4 ]*(2**15 )+arr[:,5 ]*(2**14 )+arr[:,6 ]*(2**13 )+arr[:,7 ]*(2**12 )+arr[:,8 ]*(2**11 )+arr[:,9 ]*(2**10 )+arr[:,10]*(2**9 )+arr[:,11]*(2**8 )+arr[:,12]*(2**7 )+arr[:,13]*(2**6 )+arr[:,14]*(2**5 )+arr[:,15]*(2**4 )+arr[:,16]*(2**3 )+arr[:,17]*(2**2 )+arr[:,18]*(2**1 )).reshape(arr.shape[0], 1)
    ret = np.concatenate((ret1, ret2, ret3, ret4, ret5, ret6), axis=1)
    return ret

def prec_and_recall(pred, act, totdata):
    MAX = 4
    # 
    correct = 0
    wrong = 0
    for i in range(len(pred)):
        if act[i] >= MAX:
            if pred[i] >= act[i]:
                correct += 1
            else:
                wrong += 1
    prec = correct/len(pred)

    actgood = 0
    for i in totdata:
        if i >= MAX:
            actgood += 1

    return prec, correct/actgood

if __name__ == "__main__":
    DATA = Dataset("datasets/ml-100k", "u1")
    DATA.getGenreDict()
    # 944
    prec = []
    rec = []
    for i in range(1, 100):
        #try:
        trainDF = DATA.getUserData(i, "train")
        testDF =  DATA.getUserData(i, "test")

        trainDF.to_csv("a.csv", header=None, index=False)
        testDF.to_csv("b.csv", header=None, index=False)
        tr = np.genfromtxt("a.csv", delimiter=",")
        te = np.genfromtxt("b.csv", delimiter=",")

        trainData = np.concatenate( (getEnums(tr), tr[:,-2:]), axis=1 )
        testData =  np.concatenate( (getEnums(te), te[:,-2:]), axis=1 )

        dec = DecisionTree(None, ndarray=trainData)
        dec.Config(binPercent=100, maxDepth=10)
        dec.Train()
        res = dec.Test(testData[:,:-1])
        indc = np.argsort(res)[::-1]
        for k in range(1, 20):
            topK = indc[:k]
            predRat = res[topK]
            actuRat = testData[:, -1][topK]        
            precision, recall = prec_and_recall(predRat, actuRat, testData[:, -1] )
            prec.append(precision)
            rec.append(recall)
        plt.plot(rec, prec, '.')
        plt.grid()
        plt.savefig("images/" + str(i) + ".PNG")
        

