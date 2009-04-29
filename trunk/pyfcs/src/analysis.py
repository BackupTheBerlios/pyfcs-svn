import struct
import scipy
from scipy import optimize
from scipy import weave
from scipy.weave import converters
from pylab import figure, show

from numpy import *

class AnalyseFCS:
  def __init__(self, filename="", type="fcs", dtau=1e-4, taumax=0.002):
    self.gSD = 1
    if filename <> "":
      if type == "fcs":
        self.openrawfcs(filename)
        self.autocorrelatefcs(dtau, taumax)
      elif type == "csv":
        self.opencsv()
        
  def autocorrelatefcs(self, dtau, taumax):
    ch1 = self.ch1
    dlength = len(ch1)

    if dtau < 1./self.frequency:
      dtau = 1./self.frequency
    tau = arange(0, taumax, dtau)
    dindex = int(dtau * self.frequency)
    
    g = zeros(len(tau))
    glength = len(g)
    
    expr="""
    float av1;
    float norm;
    norm = float(dlength);
    for(int i=0; i<dlength; i++){
        av1 += ch1(i);
    }
    av1 = av1/norm;
    for(int j=0; j<glength; j++){
      float gt;
      gt = 0.0;
      for(int i=0; i<(dlength-glength); i++){
        gt += ch1(i)*ch1(i+(j*dindex));
      }
      gt = gt/norm;
      g(j) = gt/(av1*av1);
    }
    """
    weave.inline(expr, ['g', 'glength', 'ch1', 'dlength', 'dindex'], type_converters=converters.blitz, compiler = 'gcc')
    g = g[1:]
    self.tau = tau[1:]
    self.g = g - g[len(g)-1]
    

  def openrawfcs(self, file):
    data = open(file, 'rb').read()
    channelMode=struct.unpack('c', data[1:2])[0] 
    # H one channel time mode, h one channel photon mode
    # X two channel time mode, x two channel photon mode
    frequency=struct.unpack('i', data[2:6])[0]
    clock=struct.unpack('i', data[6:10])[0]
    dataflag=struct.unpack('b', data[10:11])[0] 
    #Data is saved as 0: 16-bit or 1: 32-bit
    datasize = 2 # assume short int
    readflag = '>h'
    if dataflag == 1:
      datasize = 4 #32-bit->4byte increment, readflag = i
      readflag = '>i'
    ch1 = []
    ch2 = []
    for i in xrange(257, len(data)-260, datasize):
      ch1a = struct.unpack(readflag, data[i:i+datasize])
      ch1.append(ch1a[0])
    ch1 = array(ch1)
    ch2 = array(ch2)
    self.channelMode = channelMode
    self.frequency = frequency
    self.clock = clock
    self.datasize = datasize
    self.ch1 = ch1
    self.ch2 = ch2

  def savecsv(file):
    return 0

  def opencsv(data, tau, g):
    return 0  

class Corrfit:
# --------- Name conventions ---------------------------------#
#  s = p0[0]         Ratio of length of the confocal volume
#  N = p0[1]         Number of molecules in Volume
#  T = p0[2]         Ration of molecules in tripletstate
#  lifetime = p0[3]     lifetime of the tripletstate
#  Dt = p0[4]         Diffusiontimes in the confocal volume
#  a = p0[5]         abonormality constants - normally 1
#  f = p0[6]         molar fraction of different etnenities

  def __init__(self, tau, g_meas, gSD=1, triplet=0, particles=1):
    self.triplet = triplet
    self.particles = particles
    self.sucess = 0
    self.pfit = zeros(2+2*triplet+3*particles)
    self.tau = tau
    self.g_meas = g_meas
    self.gSD = gSD

  def corr3da(self, p, tau):
    s = p[0]
    N = p[1]
    T = []
    lifetimes = []
    for i in range(self.triplet):
      T.append(p[2+i])
      lifetimes.append(p[2+self.triplet+i])
    Dt = []
    f = []
    a = []
    for i in range(self.particles):
      Dt.append(p[2+2*self.triplet+i])
      f.append(p[2+2*self.triplet+self.particles+i])
      a.append(p[2+2*self.triplet+2*self.particles+i])
    #Calculate concentration factor
    cf = 1./N
    #Calculate T-factor (Tiplet-state)
    tfactor = 1
    for i in xrange(self.triplet):
      tfactor += -T[i]+T[i]*exp(-tau/lifetimes[i]) 
    #Calculate g
    g = zeros(len(tau))
    for i in range(self.particles):
      g += f[i]*(1+(tau/Dt[i])**a[i])**-1*(1+1/s**2*(tau/Dt[i])**a[i])
    return g*cf*tfactor
  
  def errfunc(self, pv, pf, fixedp, tau, g_meas, gSD):
    pv = iter(pv)
    pf = iter(pf)
    p = []
    for i in range(len(fixedp)):
        if fixedp[i]:
            ap = pf.next()
        else:
            ap = pv.next()
        p.append(ap)
    err = (self.corr3da(p, tau)-g_meas)/gSD
    return err
  
  def optimize(self, p0, fixedp):
    # fixedp is a list of len(p0) which represents if a parameter is fixed True or variable False
    pf = []
    pv = []
    for i in range(len(p0)):
      if fixedp[i]:
        pf.append(p0[i])
      else:
        pv.append(p0[i]) 
    pv, cov, info, mesg, self.success = optimize.leastsq(self.errfunc, pv[:], args=(pf, fixedp, self.tau, self.g_meas, self.gSD), full_output = 1)

    pv = iter(pv)
    pf = iter(pf)
    p = []
    for i in range(len(p0)):
      if fixedp[i]:
        p.append(pf.next())
      else:
        p.append(pv.next())
    self.pfit = p
    self.cov = cov
    return p, self.success

  def plotautocorr(self):
    fig = figure()
    ax1 = fig.add_subplot(211)
    ax1.set_xscale('log')
    ax1.set_xlim(0, max(self.tau))
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.set_position([0.1, 0.12, 0.8, 0.15])
    ax1.set_xlabel('log(tau)')
    ax1.set_ylabel('residuals')
    res = self.corr3da(self.pfit, self.tau)-self.g_meas
    ax1.plot(self.tau, res)
    ax2.set_position([0.1, 0.4, 0.8, 0.5])
    ax2.set_xscale('log')
    ax2.set_ylim((0,max(self.g_meas)))
    ax2.set_ylabel('G(tau)')
    ax2.plot(self.tau, self.corr3da(self.pfit, self.tau), "b-", self.tau, self.g_meas, "ro")
    show()
