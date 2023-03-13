import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from PIL import ImageDraw
import math
from joblib import Parallel, delayed
from threading import Thread
import random
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
    x = np.array(x/xmax*200, dtype=np.int8)
    y = np.array(y/ymax*80,dtype=np.int8)
    xmax = max(x)
    ymax = max(y)
    imgarr = np.full((xmax+10, ymax+10), 255)
    pmin = min(p)
    pmax = max(p)
    n = len(x)
    for i in range(n):
        imgarr[5+x[i]][5+y[i]] = int(255 - (p[i]-pmin)/(pmax-pmin)*255)
    print('visualize ',imgarr.shape)
    showArr(imgarr, 'visualize')

def readSign(filepath):
    #f = open('trainingSet/OnlineSignatures/Dutch/TrainingSet/Online Genuine/001_01.HWR'.'r')
    file = open(filepath)
    x = []
    y = []
    time = []
    #btn
    az=[]
    alt=[]
    p = []
    skip = True
    for line in file:
        if(not skip):
            l = line.split()
            if(time == [] or int(l[2])>time[-1]):
                x.append(int(l[0]))
                y.append(int(l[1]))
                time.append(int(l[2]))
                az.append(int(l[4]))
                alt.append(int(l[5]))
                p.append(int(l[6]))
        else:
            skip=False
    return np.array(x),np.array(y),np.array(time),np.array(az), np.array(alt), np.array(p)

def dtw(x1,x2):
    x1 = normalize(x1)
    x2 = normalize(x2)
    # kernel = np.array([1.0,2.0,1.0])
    # x1 = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, x1)
    # x2 = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, x2)
    n1 = len(x1)
    n2 = len(x2)
    mat = np.zeros((n1, n2))
    mat[0][0] = abs(x1[0]-x2[0])
    for i in range(1, n1):
        mat[i][0] = mat[i-1][0] + abs(x1[i] - x2[0])
    for i in range(1, n2):
        mat[0][i] = mat[0][i-1] + abs(x1[0] - x2[i])
    for i in range(1,n1):
        for j in range(1,n2):
            mat[i][j] = min(mat[i-1][j-1], mat[i-1][j], mat[i][j-1]) + abs(x1[i]-x2[j])
    return mat[n1-1][n2-1]

def diff(x):
    n = len(x)-1
    v = np.zeros(n)
    for i in range(n):
        v[i] = (x[i+1]-x[i])
    return v

def getv(x, time):
    n = len(x)-1
    v = np.zeros(n)
    for i in range(n):
        v[i] = (x[i+1]-x[i])/(time[i+1]-time[i])
    return v

def getacc(v, time):
    n = len(v)-1
    acc = np.zeros(n)
    for i in range(n):
        acc[i] = (v[i+1]-v[i])/(time[i+2]-time[i])
    return acc

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
    
def div(x,y):
    if(y==0):
        return 1000
    else:
        return x/y

vdiv = np.vectorize(div)

def compare(filepath1, filepath2):
    x1,y1,time1, az1, alt1, p1 = readSign(filepath1)#x, y, 1) pressure 
    x2,y2,time2, az2, alt2, p2 = readSign(filepath2)
    vx1 = ThreadWithReturnValue(target=getv, args=(x1,time1))
    vx1.start()
    vy1 = ThreadWithReturnValue(target=getv, args=(y1,time1))
    vy1.start()
    vx2 = ThreadWithReturnValue(target=getv, args=(x2,time2))
    vx2.start()
    vy2 = ThreadWithReturnValue(target=getv, args=(y2,time2))
    vy2.start()
    dp1 = ThreadWithReturnValue(target=getv, args=(p1,time1))
    dp1.start()
    dp2 = ThreadWithReturnValue(target=getv, args=(p2,time2))
    dp2.start()
    vx1 = vx1.join()
    vx2 = vx2.join()
    vy1 = vy1.join()
    vy2 = vy2.join()
    dp1 = dp1.join()
    dp2 = dp2.join()

    dpx1 = vdiv(dp1,vx1)#8) dp/dx
    dpy1 = vdiv(dp1,vy1)#9) dp/dy
    dpx2 = vdiv(dp2,vx2)
    dpy2 = vdiv(dp2,vy2)
    s1 = np.sqrt(vx1**2 + vy1**2) #4) speed
    s2 = np.sqrt(vx2**2 + vy2**2)

    
    ax1 = ThreadWithReturnValue(target=getacc, args=(vx1, time1))
    ax1.start()
    ay1 = ThreadWithReturnValue(target=getacc, args=(vy1, time1))
    ay1.start()
    ax2 = ThreadWithReturnValue(target=getacc, args=(vx2,time2))
    ax2.start()
    ay2 = ThreadWithReturnValue(target=getacc, args=(vy2,time2))
    ay2.start()
    
    ax1 = ax1.join()
    ay1 = ay1.join()
    ax2 = ax2.join()
    ay2 = ay2.join()
    

    print('Features calculated\nCalculating DTW')
    pairs = [(x1,x2), (y1,y2), (p1,p2), (az1,az2), (alt1,alt2), (vx1,vx2), (vy1, vy2), (s1,s2), (ax1,ax2), (ay1,ay2), (dp1,dp2), (dpx1,dpx2), (dpy1, dpy2)]
    threads = []
    for pair in pairs:
        threads.append(ThreadWithReturnValue(target=dtw, args=pair))
        threads[-1].start()
    res = []
    for thread in threads:
        res.append(thread.join())
    print("DTW calculated")
    visualize(x1,y1,p1)
    visualize(x2,y2,p2)
    return res


base1 = 'trainingSet/Task2/'
base2 = 'trainingSet/Task2/'

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

def generateNonMatch():
    #GENERATE 
    file = open('nonMatches.txt','w+')
    samples = []
    random.seed(0)
    for i in range(700):
        s1 = 1
        s2=1
        k1=1
        k2=1
        while( (s1==s2) or (k1==k2) or ((s1,k1,s2,k2) in samples)):
            s1 = random.randint(1,40)
            s2 = random.randint(1,40)
            if(s1>s2):
                (s1,s2) = (s2,s1)
            k1 = random.randint(1,20)
            k2 = random.randint(1,20)
            if (k1>k2):
                (k1,k2) = (k2,k1)

    for (s1,k1,s2,k2) in samples:
        dtws = compare(f'{base2}U{s1}S{k1}.TXT', f'{base2}U{s2}S{k2}.TXT')
        file.write(f'{base2}U{s1}S{k1},{base2}U{s2}S{k2},{dtws}')
    
    file.close()


    
def generateGenuine():
    #GENERATING 1000 GENUINE MATCHES
    f = open('allmatches.txt','w+')
    random.seed(1000)
    for i in range(1,41):
        samples = []
        for j in range(25):
            k1 = 1
            k2 = 1
            while (k1==k2) or ((k1,k2) in samples):
                k1 = random.randint(1,20)
                k2 = random.randint(1,20)
                if (k1>k2):
                    (k1,k2) = (k2,k1)
        for (k1,k2) in samples:
            dtws = compare(f'{base2}U{i}S{k1}.TXT', f'{base2}U{i}S{k1}.TXT')
            f.write(f'U{i}S{k1},U{i}S{k2},{dtws}')
    f.close()

def generateForgeryPairs():
    #GENERATE 320 imposter pairs
    random.seed(2000)
    file = open('forgery_pairs.txt')
    for i in range(1,41):
        for j in range(1,9):
            dtw = compare(f'{base2}U{i}S{j}.TXT', f'{base2}U{i}S{j+20}.TXT')
            file.write(f'U{i}S{j},U{i}S{j+20},{dtw}')
    file.close()
            
#generateForgeryPairs()
'''
Observations - 
dx, dy is giving highest seperation
dp is not preferrable
'''
print(compare(f'{base2}U1S1.TXT', f'{base2}U2S2.TXT'))