import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dst
import time

def t6hat(rmax,rmin,x):

   xx   = np.abs(x)
   ones = np.ones_like(xx)
   xmax = rmax*ones
   xmin = rmin*ones

   xx1 = np.minimum(xx,xmax)
   xx2 = np.maximum(xx1,xmin)
   xx3 = (ones - (xx2 - rmin)/(rmax-rmin))*math.pi
   xx4 = sin6hat(xx3)
   return xx4

def sin6hat(x):
    res = np.sin(6.0*x)/3.0 - 3.0*np.sin(4.0*x)   + 15.0*np.sin(2.0*x) - 20.0*x
    res = -res/(20.0*math.pi)
    return res

def doublefilt(zerocut,onecut,lowzerocut,lowonecut,input):
    ntotal = len(input)

    z =  np.fft.fftshift(np.fft.fft(input))
    fr = np.zeros_like(input)

    for jj in range(ntotal):
        fr[jj] = (jj - ntotal/2 )/ (ntotal/2)

    hat  = t6hat(zerocut,onecut,fr)

    # z = hat*z


    if(lowonecut > 0.0001):
        hat1 = np.ones_like(fr)
        hat1 = hat1 - t6hat(lowonecut,lowzerocut,fr)
        hat = hat*hat1

    z = hat*z

    #plt.plot(hat1*hat)
    #plt.show()

    z =  np.fft.fftshift(z)
    out = np.fft.ifft(z)
    out = np.real(out)

    return out, hat

def triplefilt(zerocut,onecut,lowzerocut,lowonecut,input,shapefft):
    ntotal = len(input)

    z =  np.fft.fftshift(np.fft.fft(input))

    z = z*shapefft

    fr = np.zeros_like(input)

    for jj in range(ntotal):
        fr[jj] = (jj - ntotal/2 )/ (ntotal/2)

    hat  = t6hat(zerocut,onecut,fr)

    # z = hat*z


    if(lowonecut > 0.0001):
        hat1 = np.ones_like(fr)
        hat1 = hat1 - t6hat(lowonecut,lowzerocut,fr)
        hat = hat*hat1

    z = hat*z

    #plt.plot(hat1*hat)
    #plt.show()

    z =  np.fft.fftshift(z)
    out = np.fft.ifft(z)
    out = np.real(out)

    return out, hat


def mydst2(input):

    start = time.time()
    nd1, nd2 = np.shape(input)
    temp   = dst(input[1:nd1-1,1:nd1-1], type=1, axis = 0, norm='ortho')

    ndd1, ndd2 = np.shape(temp)
    temp1  = dst(temp,  type=1, axis = 1, norm='ortho')


    output = np.zeros((nd1,nd2))
    output[1:nd1-1,1:nd2-1] = temp1

    finish = time.time()

    print('MyDST2: ',ndd1,ndd2,'ook ','{:05.2f}'.format(finish-start),' sec')

    return output

def my_search(x,hsize,ndim):

   shx = 2.*hsize/(ndim-1)

   rnx =  (x + hsize)/ shx
   nx = rnx.astype(int)
   dx = x + hsize - nx*shx
   a2 = dx/shx

   good = np.ones(np.shape(x), dtype='bool')

   good[ (nx == 0) & (a2 < -.05)] = False
   good[ (nx < 0) ]               = False

   good[ (nx == ndim-1) & (a2 > 0.05)] = False
   good[ nx > ndim -1 ]               = False


   nx[nx < 0] = 0
   nx[nx > ndim-2] = ndim-2

   dx = x + hsize - nx*shx         # this is a repetion of the line above, it is needed for the case nx = ndim-1
   a2 = dx/shx

   a1 = 1.0 - a2

   return a1,a2,nx,good


def my_search_from_zero(x,hsize,ndim):

   shx = hsize/(ndim-1)

   rnx =  x/shx
   nx = rnx.astype(int)
   dx = x - nx*shx
   a2 = dx/shx

   good = np.ones(np.shape(x), dtype='bool')

   good[ (nx == 0) & (a2 < -.05)] = False
   good[ (nx < 0) ]               = False

   good[ (nx == ndim-1) & (a2 > 0.05)] = False
   good[ nx > ndim -1 ]               = False


   nx[nx < 0] = 0
   nx[nx > ndim-2] = ndim-2

   dx = x  - nx*shx         # this is a repetion of the line above, it is needed for the case nx = ndim-1
   a2 = dx/shx

   a1 = 1.0 - a2

   return a1,a2,nx,good


def my_bilinear_intrp (data,hsizex,hsizey,xcoor,ycoor):
    (ndimy,ndimx) = np.shape(data)
#    print(ndimx,ndimy)

    a1x,a2x,nx,goodx =  my_search(xcoor,hsizex,ndimx)
    a1y,a2y,ny,goody =  my_search(ycoor,hsizey,ndimy)

    res = a1y*(a1x*data[ny,nx]  + a2x*data[ny,nx+1])  + a2y*(a1x*data[ny+1,nx]  + a2x*data[ny+1,nx+1])
    res [ (goodx == False) | (goody == False) ] = 0.0

    return res

def my_bilinear_intrp_from_zero (data,hsizex,hsizey,xcoor,ycoor):
    (ndimy,ndimx) = np.shape(data)
#    print(ndimx,ndimy)

    a1x,a2x,nx,goodx =  my_search_from_zero(xcoor,hsizex,ndimx)
    a1y,a2y,ny,goody =  my_search_from_zero(ycoor,hsizey,ndimy)

    res = a1y*(a1x*data[ny,nx]  + a2x*data[ny,nx+1])  + a2y*(a1x*data[ny+1,nx]  + a2x*data[ny+1,nx+1])
    res [ (goodx == False) | (goody == False) ] = 0.0

    return res

def my_quadrature(asy_step,length,n,b):

   bn = b**n
   large_number = round(2. * length/asy_step)

   tryarr = np.array(range(large_number))

   eta = asy_step*np.array(range(large_number))

   etan = eta**n

   xxx = eta*etan / (bn+ etan)

   nodes = np.argmax(xxx > length)

   # print(nodes,xxx[nodes])

   xxx = xxx[0:nodes]
   etan = etan[0:nodes]

   rjacob = etan*(bn*(n+1)+etan)/( (bn+etan)*(bn+etan))

   return xxx,rjacob,nodes


