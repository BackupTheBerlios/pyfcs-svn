#reload(analysis)
import analysis
from scipy import optimize
from scipy.integrate import simps, trapz
from numpy import ones, zeros, arange, diff, log, array
import pylab as p

class Distri:
    
    def __init__(self, N, NSD, s, tau, g_meas, gSD):
        self.tau = tau
        self.g_meas = g_meas
        self.gSD = gSD
        self.s = s
        self.N = N

    def gsfunc(self, c, tauD, tau):
        return c*(1.+tau/tauD)**-1. * (1.+tau/(self.s**2.*tauD))
        
    def gfunc(self, c, tauD, tau):
        gre = zeros(len(tau))
        for i in range(len(c)):
            gre += self.gsfunc(c[i], tauD[i], tau)
        return gre
    
    def maxent(self, tauDmin, tauDmax, M, alpha):
        return False
    
    def starchev(self, tauDmin, tauDmax, M, lagr=[0.0, 0.0], fixl=[True, True]):
    #Konstantin Starchev, Jaques Buffle, and Elias Perez
    #Application of Fluorescence Correlation Spectroscopy: Polydispersity Measurements
    #Journal of Colloid and Interface Science 213, 479-487 (1999)
    
        def minfunc(c, lagr, fixl, tauD, tau, g_meas, gSD):
            ret = []
            #Lagrangeparamter aus Fitparameter oder Paramterliste
            c = c.tolist()
            laa = [] 
            for i in range(len(fixl)):
                if fixl[i]:
                    laa.append(lagr[i])
                else:
                    laa.append(c.pop())
                    lam = laa[0]
            lar = laa[1]
            c = array(c)
            #Berechnen von Normierungskonstante und Delta tauD
            t1 = min(tau)
            t2 = max(tau)
            norm = 1./(t2-t1)
            dtD = tauD[1]-tauD[0]
            #Randbedingung c[i] muss groesser 0 sein
            #for i in range(len(c)):
            #    if c[i]<0:
            #        c[i]=0
            res = 0
            #Durchlaufen von c-Vektor, Enden werden nicht beruecksichtig wg. Ableitung
            for i in range(2, len(c)-2, 1):
                #Berechung von Vektor fuer Integration
                g1 = g_meas - self.gfunc(c, tauD, tau)
                g2 = self.gsfunc(1, tauD[i], tau)
                gi = -2./gSD**2 * g1 *g2
                a1 = norm * simps(gi, self.tau)
                a2 = 2.*lam*(sum(c)/self.N**2-1./self.N)
                a3 = 2.*lar/dtD**4*(c[i-2]-4*c[i-1]+6*c[i]-4*c[i+1]+c[i+2]) #okay
                res = a1+a2+a3
                ret.append(res)
            #Durch Ableitung verliert man je 2 Werte an den Enden -> auffuellen
            ret.insert(0, ret[0])
            ret.insert(0, ret[1])
            ret.append(res)
            ret.append(res)
            #Ableitungen von Lambdas anhaengen falls es frei Variablen sind
            #lam
            if fixl[0]:
                dlam = (1-sum(c)/self.N)**2
                ret.append(dlam)
            elif fixl[1]:
                dlar = 1./dtD**2 * sum(diff(c, n=2))
                ret.append(dlar)
            #Anhaengen der Lagrangeparameter an die Liste
            for i in range(len(lagr)):
                if not fixl[i]:
                    ret.append(laa[i])
            return ret
    
        #c0 = self.N/M*ones(M)
        
        c0 = zeros(M)
        #Append Lagrange parameters to parameter list for leastsq-fit 
        c0 = c0.tolist()
        for i in range(len(lagr)):
            if not fixl[i]:
                c0.append(lagr[i])
        c0 = array(c0)
        #Create array of TauD values for the histogram
        tauD = arange(tauDmin, tauDmax, (tauDmax-tauDmin)/M)
        self.tauD = tauD
        #Fiting of the data values with leastq
        #Problem: leastsq without constraints -> The constraint c>0 is not fulfilled
        self.fit, self.err = optimize.leastsq(minfunc, c0[:], args=(lagr, fixl, tauD, self.tau, self.g_meas, self.gSD))
        #Aufspalten der gefitteten Parameter in Lagrange-Parameter und c-Werte
        self.lagrange = []
        fit = self.fit.tolist()
        for i in range(len(fixl)):
            if fixl[i]:
                self.lagrange.append(lagr[i])
            else:
                self.lagrange.append(fit.pop())
        self.c = fit

#autocor = analysis.AnalyseFCS("/home/thomas/100.58nM_1.fcs", "fcs", 1e-5, 0.01)
autocor = analysis.AnalyseFCS("C:\\temp\\C1-2.fcs", "fcs", 1e-5, 0.002)
#corf = analysis.Corrfit(autocor.tau, autocor.g, autocor.SD, 1, 1)
#p = [s, Number of Particles, Fractions in Triplet state, Tiplet-states lifetimes, Diffusion times, Fractions, alpha]
#p0 = [1.5, 14., 0.1, 1e-6, 1e-3, 1., 1.]
#fv = Vector of fixed values 1=fixed 0=free
#fv = [0, 1, 0, 0, 0, 1, 1]
#corf.optimize(p0, fv)
#corf.plotautocorr()
distrf = Distri(0.3, 0.05, 11.6, autocor.tau, autocor.g, autocor.gSD) #N, NSD, s, tau, g, gSD
distrf.starchev(1.e-6, 20.4e-6, 30, [1.e-9, 1.e-9], [False, False])#Nr.0 = lam, Nr. 1 = lar
#p.plot(log(autocor.tau), autocor.g)
p.plot(distrf.tauD, distrf.c)
print 'Fit successful: %i' % distrf.err
print 'Lambda-R: %s' % distrf.lagrange[1]
print 'Lambda-M: %s' % distrf.lagrange[0]
p.show()
#print distrf.err
