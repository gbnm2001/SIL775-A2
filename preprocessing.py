import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from PIL import ImageDraw
import math
from joblib import Parallel, delayed
from threading import Thread
import random

def showArr(image_arr, name, points = [], points2 = []):
    image = im.fromarray(image_arr)
    rgbimg = im.new("RGBA", image.size)
    rgbimg.paste(image)
    draw = ImageDraw.Draw(rgbimg)
    for point in points:
        try:
            draw.rectangle((point[1],point[0],point[1]+5,point[0]+5), fill= (255,0,0,128))
        except:
            pass
    for point in points2:
        try:
            draw.rectangle((point[1],point[0],point[1]+5,point[0]+5), fill= (0,0,255,128))
        except:
            pass
    rgbimg.show(title=name)


def normalize(arr):
    com = np.mean(arr)
    sd = math.sqrt(np.var(arr))
    def norm(x):
        x = (x-com)/sd
        return x
    vn = np.vectorize(norm)
    arr = vn(arr)
    return arr

#Given is the x,y coordinate and the pen pressure
#Calculate the vx,vy also
#Do matching that's it
#Create a visualization function that plots the online image

def scale(x):
    xmin = min(x)
    x = x-xmin
    xmax = max(x)
    x = x*100
    x = x.astype(int)
    return x

def visualize(y,x,p):
    x = scale(x)
    y = scale(y)
    xmax = max(x)
    ymax = max(y)
    imgarr = np.full((xmax+10, ymax+10), 255)
    pmin = min(p)
    pmax = max(p)
    n = len(x)
    for i in range(n):
        imgarr[5+x[i]][5+y[i]] = int(255 - (p[i]-pmin)/(pmax-pmin)*255)
    showArr(imgarr, 'visualize')

def readSign(filepath):
    #f = open('trainingSet/OnlineSignatures/Dutch/TrainingSet/Online Genuine/001_01.HWR'.'r')
    file = open(filepath)
    x = []
    y = []
    p = []
    for line in file:
        l = line.split()
        x.append(int(l[0]))
        y.append(int(l[1]))
        p.append(int(l[2]))
    x = normalize(np.array(x))
    y = normalize(np.array(y))
    p = normalize(np.array(p))
    return x,y,p

def dtw(x1,x2):
    mat = np.zeros((len(x1), len(x2)))
    mat[0][0] = abs(x1[0]-x2[0])
    for i in range(1, len(x1)):
        mat[i][0] = mat[i-1][0] + abs(x1[i] - x2[0])
    for i in range(1, len(x2)):
        mat[0][i] = mat[0][i-1] + abs(x1[0] - x2[i])
    for i in range(len(x1)):
        for j in range(len(x2)):
            mat[i][j] = min(mat[i-1][j-1], mat[i-1][j], mat[i][j-1]) + abs(x1[i]-x2[j])
    return mat[len(x1)-1][len(x2)-1]

def getv(x):
    v = np.zeros(len(x)-1)
    n = len(x)-1
    for i in range(n):
        v[i] = (x[i+1]-x[i])
    v = normalize(v)
    return v

def getSinCos(vx,vy):
    sin = np.zeros(len(vx))
    cos = np.zeros(len(vx))
    for i in range(len(vx)):
        a = math.atan2(vy[i], vx[i])
        sin[i] = math.sin(a)
        cos[i] = math.cos(a)
    normalize(sin)
    normalize(cos)
    return sin,cos
    

def compare(filepath1, filepath2):
    x1,y1,p1 = readSign(filepath1)#x, y, 1) pressure 
    x2,y2,p2 = readSign(filepath2)
    vx1 = getv(x1)#2) vx
    vy1 = getv(y1)#3) vy
    vx2 = getv(x2)
    vy2 = getv(y2)
    s1 = np.sqrt(vx1**2 + vy1**2) #4) speed
    s2 = np.sqrt(vx2**2 + vy2**2)
    ax1 = getv(vx1)#5) ax
    ay1 = getv(vy1)#6) ay
    ax2 = getv(vx2)
    ay2 = getv(vy2)
    dp1 = getv(p1)#7) dp
    dp2 = getv(p2)
    dpx1 = dp1/vx1#8) dp/dx
    dpy1 = dp1/vy1#9) dp/dy
    dpx2 = dp2/vx2
    dpy2 = dp2/vy2
    
    return (dtw(x1,x2), dtw(y1,y2), dtw(p1,p2), dtw(vx1,vx2), dtw(vy1, vy2),dtw(s1,s2), dtw(ax1,ax2), dtw(ay1, ay2), dtw(p1,p2), dtw(dpx1, dpx2), dtw(dpy1, dpy2))


base1 = 'trainingSet/OnlineSignatures/Dutch/TrainingSet/Online Forgeries/'
base2 = 'trainingSet/OnlineSignatures/Dutch/TrainingSet/Online Genuine/'

#print(compare(f'{base2}001_6.HWR', f'{base2}001_7.HWR'))

def runAllGenuineParallel():
    f = open('allmatches3.txt','w+')
    slist = [1,2,3,4,6,9,12,14,15,16]
    for i in slist:
        sample = f'00{i}'[-3:]
        for j in range(6,12):
            out = Parallel(n_jobs=16)(delayed(compare)(f'{base2}{sample}_{j}.HWR', f'{base2}{sample}_{k}.HWR') for k in range(j+1,18))
            for k in range(j+1,18):
                f.write(str(f'{sample}_{j},{sample}_{k},{out[k-j-1]}\n'))

def runNonmatchParallel():
    file = open('nonMatches.txt','w+')
    slist = [1,2,3,4,6,9,12,14,15,16]
    samples = []
    random.seed(0)
    for j in range(1000):
        s1 = 1
        s2 = 1
        k1 = 1
        k2 = 1
        while ((s1==s2) or (k1==k2) or (s1,k1,s2,k2) in samples):
            s1 = slist[random.randint(0,len(slist)-1)]
            s2 = slist[random.randint(0,len(slist)-1)]
            k1 = random.randint(1,24)
            k2 = random.randint(1,24)
            if(s1>s2):
                (s1,s2) = (s2,s1)
            if(k1>k2):
                (k1,k2) = (k2,k1)
        samples.append((s1,k1,s2,k2))
    out = Parallel(n_jobs=8)(delayed(compare)(f"{base2}{f'00{s1}'[-3:]}_{k1}.HWR", f"{base2}{f'00{s2}'[-3:]}_{k2}.HWR") for (s1,k1,s2,k2) in samples)
    count = 0
    for (s1,k1,s2,k2) in samples:
        file.write(f"{f'00{s1}'[-3:]}_{k1},{f'00{s2}'[-3:]}_{k2},{out[count]}\n")
        count += 1
    file.close()

class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
def runAllGenuine():
    f = open('allmatches.txt','w+')
    for i in range(1,17):
        sample = f'00{i}'[-3:]
        for j in range(1,24):
            threads = []
            for k in range(j+1,25):
                threads.append(ThreadWithReturnValue(target = compare, args = (f'{base2}{sample}_{j}.HWR', f'{base2}{sample}_{k}.HWR')))
                threads[-1].start()
            out = []
            for thread in threads:
                out.append(thread.join())
            for tup in out:
                f.write(str(f'{tup}\n'))


#runNonmatchParallel()

'''
Observations - 
dx, dy is giving highest seperation
dp is not preferrable

'''
