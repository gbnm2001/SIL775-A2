from preprocessing import *
import sys
s = '9.68138776e+00 3.52676049e-01 5.88078954e-02 1.22593685e+00 1.23058221e-01 9.04712780e-02 1.31380209e-01 7.21000000e+02'
s = list(map(float, s.split()))
w = np.array(s[:-1])
print(sys.argv)
if(len(sys.argv) != 3):
    print(f"Usage : python matcher.py sign1_path sign2_path")
else:
    dtw = compare(sys.argv[1], sys.argv[2])
    dtw = np.array(dtw)
    dtw = dtw*w
    if(dtw.sum() > s[-1]):
        print('False')
    else:
        print('True')
