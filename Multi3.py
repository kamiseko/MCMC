

#!/usr/bin/python3



### 1. Data import

import sys                                      ## system
import numpy as np                              ## Matrix Calculate
import glob                                     ## global variable
import random                                   ## random sample
from multiprocessing import Pool                ## palarellel
from multiprocessing import Array,Pipe
from math import exp,gamma,log,sqrt             ## exp,log,sqrt cal
from scipy.stats import multivariate_normal
from numpy import inf
import scipy.linalg as sl
import scipy.stats as ss
import time
from sklearn.decomposition import FactorAnalysis
from sklearn import linear_model
import multiprocessing
import ctypes


if len(sys.argv)>1:
    list_of_files = sorted(glob.glob('/stfs1/uhome/y90027/MCMC/carb/demand/*.csv'))
    list_of_files2 = sorted(glob.glob('/stfs1/uhome/y90027/MCMC/carb/prices/*.csv'))
else:
    import matplotlib.pyplot as plt
    #list_of_files = sorted(glob.glob('D:\OneDrive\dat\carb\demand\*.csv'))
    #list_of_files2 = sorted(glob.glob('D:\OneDrive\dat\carb\prices\*.csv'))
    list_of_files = sorted(glob.glob('D:\OneDrive\dat\yogurt\demand\*.csv'))
    list_of_files2 = sorted(glob.glob('D:\OneDrive\dat\yogurt\prices\*.csv'))




from functools import partial
import time
#print(list_of_files)
demand = [np.loadtxt( f,skiprows=1,delimiter="," ) for f in list_of_files]
prices = [np.loadtxt( f,skiprows=1,delimiter="," ) for f in list_of_files2]
#print(demand)

#####################  2.  Fucntions
###prior : 0 = pu0 , 1 = pv0, 2 = gu0 , 3 = gv0 , 4 = tpp , 5 = tpg
def mdcev(psid,gamd,dem,pri):
    x = np.array(dem)
    p = np.array(pri)

    k = 35
    n = 1
    expgam = np.array(np.exp(gamd))
    temp = psid
    temp2 = expgam

    v = temp - np.log(np.array(x*temp2+1)) - np.log(p)
    ll = np.zeros((n,1))
    xp = x >0
    for i in range(n):
        vi = v
        xi = x
        mv = xp.sum()  ## 0 = col  1 = row
   #print("v",v)
    #print(mv)
        lli = (vi[xi>0]).sum() - mv * np.log(np.exp(vi).sum() ) + np.log(gamma(mv))
        gamj = expgam[xi>0]
        xij = xi[xi>0]
        jacv =  gamj/(gamj * xij  + 1)  ##jacobian
        ll[i] = lli + np.log(jacv).sum() + np.log( (1/jacv).sum() )
    sumll = sum(ll)
    #print("sumll",sumll)
    return(sumll)




def rwmh(d,p,x,ll,alpha,f,psi,g,kt,r,betai,zphi,bh,gh,abh):     ## prior[0] =   priorh = {"zt":zt[i, :], "vp":vp, "zp":zp[i, :], "vg": vg,#"tpp":tpp[i] * np.diag(np.array([1] * kp)),#"tpg":tpg[i] * np.diag(np.array([1] * k)) }
    global vp,vg,v
    k = d.shape[1] ###
    kp = k - 1
    #new_par = par
    #print(new_par)
    af = f.dot(gh.transpose())
    bg = g.dot(bh[0:kp,:].transpose())
    n = psi.shape[0]
    ite = np.zeros((n,2))                         ### num of mcmc iteration
    ## sampling psi
    for i in range(n):

        llb = ll[i] + multivariate_normal.logpdf(alpha[i,:],af[i,:],sigma)
        acc = 0
        count = 0
        init = psi
        while acc == 0  :
            # print("here")
            count += 1
            ite[i,0] += 1
            nalpha = multivariate_normal.rvs(alpha[i,:].transpose(),tpa).transpose()
            nll = mdcev(psi[i,:].transpose(),nalpha,d[i,:],p[i,:])
            nllb = nll + multivariate_normal.logpdf(nalpha.transpose(),af[i,:],sigma)
            if nllb - llb> np.log(random.uniform(0,1)):
                acc = 1
                alpha[i,:] = nalpha.transpose()
                ll[i] = nll
            else:
                acc = 1
        llb = ll[i] + multivariate_normal.logpdf(psi[i,0:(kp)],bg[i,:],delta)
        ca = nalpha
        acc = 0
        while acc == 0 :
            ite[i,1] = ite[i,1]+1
            npsi = np.concatenate((np.array(multivariate_normal.rvs(psi[i,0:(kp)].transpose(),tpp).transpose()),[0]))
            # print(new_par["psi"].shape)
            nll = mdcev(npsi,ca,d[i,:],p[i,:])
            #print(nll)
            nllb = nll + multivariate_normal.logpdf(npsi[0:(kp)].transpose(),bg[i,:],delta)
            if nllb - llb > np.log(random.uniform(0,1)) :
                acc = 1
                psi[i,:] = npsi.transpose()
                ll[i] = nll
            else:
                acc = 1

    ### f & g
    gmat = np.zeros((kl,kl))
    gmat[0, 0] = 1
    gmat[1:(kl ), 0] = -betai
    w = np.diag([1.] * 3)
    #print("gmat",gmat.shape)
    result = kalmanFilter(np.concatenate((alpha,psi[:,0:kp]),axis=1), abh, sdm, gmat, w, m0, c0, kt, r)
    mt = result["mt"]
    ct = result["ct"]
    #print( xpnd(np.array([1.,2.,3.,4.,5.,6.])))
    #ct = ct.reshape((len(ct),1))
    kt = result["kt"]
    #print(ct)
    theta = BackSample(mt,ct,gmat,w,kt)
    f = theta[:,0:(1)]
    g = theta[:,1:(1+2)]
    ## beta
    #print(f.shape)
    fii = np.concatenate((np.array([0]).reshape((1,1)),f[0:(f.shape[0]-1),:]))
    xf =-fii[kt == 1]
    for j in range(2):
        yg = g[kt==1,j]
        if yg is not None:
            bs = np.linalg.inv(vbi[j,j]+xf.transpose().dot(xf)/vg[j,j] )
            bm = bs.dot( vbi[j,j] * zphi[j] + xf.transpose().dot(yg)/vg[j,j] )
        else:
            bs = np.linalg.inv(vbi[j, j] )
            bm = bs.dot(vbi[j, j] * zphi[j])
        betai[j] = chol(bs).transpose().dot(multivariate_normal.rvs(0,1))  + bm
    return({"gam":alpha,"f":f,"psi":psi,"g":g,"ll":ll,"ite":ite,"kt":kt,"betai":betai.transpose()})

def EFAs(alpha,f,psi,g):
    ## ah
    for i in range(k):
        if i == 0:
            gh[0,0] = 1
        elif i<=0:
            b2d = alpha[:,i] - f[:,i]
            As = np.linalg.inv(gv0i[0:(i-1),0:(i-1)])
        else:
            As = np.linalg.inv(gv0i + np.array(sigmai[i,i]).dot(f.transpose()).dot(f) )
            am = As.dot( gv0i.dot(gu0[i,:]) + np.array(sigmai[i,i]).dot(f.transpose()).dot(alpha[:,i]) )
            gh[i,:] =(np.array(chol(As)).reshape((1,len(chol(As)))).transpose().dot(np.array(multivariate_normal.rvs(([0.]*1),1 )).reshape((1,1)) )+am ).transpose()
    ###
    for i in range(kp):
        if i == 0:
            bh[0, 0] = 1
        elif i <= 1:
            b2d = psi[:, i] - g[:, i]
            bs = np.linalg.inv(bv0i[0:i, 0:i] + np.array(deltai[i,i])*((g[:,0:i].transpose()).dot(g[:,0:i]) ))
            bm = bs.dot( (bv0i[0:i, 0:i].dot(bu0[i,0:i].transpose())+ deltai[i,i]*(g[:,0:i].transpose().dot(b2d)) ) )
            bh[i,0:i] = (np.array(chol(bs)).reshape((1,len(chol(bs)))).transpose().dot(np.array(multivariate_normal.rvs(([0.]*(1)),1 )).reshape((1,1))  )+bm ).transpose()

        else:
            bs = np.linalg.inv(bv0i + deltai[i, i]*(g.transpose().dot(g)))
            bm = bs.dot(bv0i.dot(bu0[i, :]) + deltai[i, i]*(g.transpose().dot(psi[:, i])) )
            bm = bm.reshape((2,1))
            bh[i,:] = (np.array(chol(bs)).transpose().dot(np.array(multivariate_normal.rvs(([0.] * 2), 1)).reshape((2,1))) + bm).transpose()
    return({"gh":gh,"bh":bh})

def rwishart(nu,v):
    try:
        m = v.shape[0]
    except:
        m = 1
    df = int((nu + nu - m + 1) - (nu -m+1))
    if m > 1:
        #t = np.diag( sqrt( ss.chi2.rvs(df=df, size=np.array([1] * m) )  )  )
        t = np.diag(np.sqrt(ss.chi2.rvs(df=df, size=m)))
        l = np.tril_indices(t.shape[0], -1)

        t[l] = np.random.normal(0,1,(m * (m + 1)/2 - m ) )
    else:
        t = sqrt( ss.chi2.rvs(size=1,df=df) )
    if v.shape[0] == 1:
        u = np.sqrt(v)
    else:
        u = np.linalg.cholesky(v)
    c = np.dot(t.transpose(), u)
    ci = np.array(sl.solve_triangular(c,np.diag([1]*m),lower=False))
    #print(np.dot(ci.transpose(),ci))
    return{
        "W":np.dot(c.transpose(),c),
        "IW":np.dot(ci.transpose(),ci),
        "C":c,
        "CI":ci
    }

def chol(x):
    if(x.shape[1] == 1 and x.shape[0]==1):
        x = np.array(np.sqrt(x)).reshape((1,1))
    else:
         x = np.linalg.cholesky(x)
    return(x)

def xpnd(x):
    dim = int( (-1 + sqrt(1 + 8 * len(x)))/2 )
    new = np.zeros((dim,dim))
    inds = np.triu_indices_from(new)
    new[inds] = x
    new[(inds[1],inds[0])] = x
    return new

def rmultireg(y,x,bbar,a,nu,v,n):
    l1 = y.shape[0]
    m  = y.shape[1]
    k  = x.shape[1]
    ## first draw sigma
    #print(m)
    ra = chol(a)
    w  = np.vstack((x, ra))
    #print("bbar:",bbar)
    z  = np.vstack((y, np.dot(ra, bbar) ))
    #print("z:",z)
    #print(np.dot(w.transpose(),z))
    #ir = 1 / chol(chol(np.dot(w.transpose(), w)), np.diag([1] * k))
    #print("alkfj",chol(np.dot(w.transpose(),w)))
    #ir = 1/ chol(chol(np.dot(w.transpose(),w)))
    ir = sl.solve_triangular(chol(np.dot(w.transpose(), w)),np.diag([1]*k))
    btilde = np.dot( np.dot(ir, ir.transpose()), np.dot(w.transpose(),z))
    #print("ir", ir.shape)
    #print("btilde",btilde.shape)
    temp = z - np.dot(w,btilde)
    s  = np.dot(temp.transpose(),temp)
    out1 = np.array([0.]*n*m*k).reshape((n,m*k))
    if m == 1 :
        out2 = np.array([0.]*n).reshape((n,1))
    if m >  1 :
        out2 = {"list": [] for i in range(n)}
    for i in range(n):
        #rwout = rwishart(nu+l1, np.linalg.inv( np.dot(chol(v+s).transpose(), chol(v+s))  ))
        rwout = rwishart(nu + l1, np.linalg.inv(v+s))
        #print(np.random.standard_normal(m*k))
        out1[i,:] = (btilde + np.dot(np.dot(ir, np.random.standard_normal(m*k).reshape((k,m)) ), rwout["CI"].transpose() ) ).reshape((1,m*k))
        #print(out1[i,:])
        if m == 1:
            out2[i,0] = np.array(rwout["IW"])
        else:
            out2[i] = np.array(rwout["IW"])
    return {"B":out1, "Sigma":out2, "BStar":btilde}


def bmreg(y,x,theta,Lambda,u0,v0,f0,g0):
    n = y.shape[0]
    m = y.shape[1]
    k = x.shape[1]
    l = m * k
    try:
        lambdai = np.linalg.inv(Lambda )
    except :
        lambdai = 1/Lambda

    v0i = np.linalg.inv(v0)

    ## generate theta
    var = np.linalg.inv(x.transpose().dot(x)*lambdai+v0i)
    #print(var.shape,y.shape,x.shape,lambdai.shape,v0i.shape,u0.shape)
    mean = var.dot((x.transpose()*lambdai).dot( y )+ v0i.dot(u0) )

    s_theta = multivariate_normal.rvs(mean,var).reshape((k,m))

    ## generate lambda
    res = y - x.dot(s_theta)
    gn = np.linalg.inv(res.transpose().dot(res)+np.linalg.inv(g0) )

    #s_lambdai = rwishart(gn,f0+n)
    s_lambdai = ss.invwishart.rvs(f0+n,gn)
    #print(s_lambdai)
    try:
        s_lambda = np.linalg.inv(s_lambdai)
    except:
        s_lambda = 1/s_lambdai

    return({"s_theta":s_theta,"s_lambda":s_lambda})


def kalmanFilter(data,F,V,G,W,m0,c0,kt,r):
    t = data.shape[0]
    m = data.shape[1]
    p = F.shape[1]
    m = int(p * (p + 1)/2)
    #print(kt)
    mt = np.zeros((t,p))
    ct = np.zeros((t,m))

    m_t = m0
    Ct = c0
    k_t = 1
    kt = kt.reshape((kt.shape[0]))
    #if G[1,0] < 0:
    #    G[1,0] = 0
    #if G[2,0] < 0:
    #    G[2,0] = 0
    for i in range(t):
        Ft = F
        Gt = G
        Wt = W
        kt[i] = k_t
        if k_t == 1:
            Gt = Gt
        else:
            Gt[1:(1+2),0:(1)] = np.zeros((2,1))
            Wt[1:(1+2),(1):(1+2)] = np.zeros((2,2))

        ######################################################
        ### predict theta t|y-1
        at = Gt.dot(m_t)
        Rt = np.dot(np.dot (Gt,Ct) , Gt.transpose()) + Wt

        sft = Ft.dot(at)
        Qti = np.linalg.inv(np.dot(np.dot(Ft,Rt),Ft.transpose() ) + V)

        m_t = at + Rt.dot(Ft.transpose()).dot(Qti).dot(data[i,:].reshape((1,cmn)).transpose()-sft)
        Ct = Rt - Rt.dot(Ft.transpose()).dot(Qti).dot(Ft).dot(Rt)
        mt[i,:] = m_t.transpose()
        #print("Rt",Rt)
        #print("at",at)
        #print("Gt",Gt)

        #print(Ct[np.tril_indices(Ct.shape[0], 0)])
        ct[i,:] = Ct[np.tril_indices(Ct.shape[0], 0)]
        if m_t[0] < r:
            k_t = 1
        else:
            k_t = 0
    return({"mt":mt,"ct":ct,"kt":kt} )


def BackSample(mt,ct,G,W,kt):
    t = mt.shape[0]
    p = mt.shape[1]
    #print("t",t)
    st = np.zeros((t,p))

    sh_t = mt[t-1,:].reshape((1,3)).transpose()
    H_t = xpnd(ct[t-1,:].transpose())

    for i in range(t-1,0,-1):
        #H_t[1:3, 1:3] = 0
        if H_t[1,1] == 0:
            s_t = np.sqrt(H_t).transpose().dot( (multivariate_normal.rvs(([0]*p),1)).reshape((p,1)) )+sh_t
        else:
            s_t = chol(H_t).transpose().dot( (multivariate_normal.rvs(([0]*p),1) ).reshape((p,1))) +sh_t

        #print(s_t)
        st[i,:] = s_t.transpose()
        if i != 0:
            G_t = G
            W_t = W
            if kt[i] == 1:
                G_t = G_t
            else:
                G_t = G_t[0:1, :]
                W_t = W_t[0:1, 0:1]
                s_t = s_t[0:1, :]
            m_t = mt[i-1,:].reshape((1,3)).transpose()
            C_t = xpnd(ct[i-1,:].transpose())

            a_t =  G_t.dot(m_t)
            try:
                #print("WT",W_t)
                #print(np.dot(np.dot(G_t, C_t), G_t.transpose()) + W_t)
                R_ti = np.linalg.inv(np.dot(np.dot(G_t,C_t),G_t.transpose()) + W_t)
                #print(R_ti)
            except:
                #print("WT",W_t)
                #print(np.dot(np.dot(G_t,C_t),G_t.transpose()) + W_t)
                R_ti = 1/(np.dot(np.dot(G_t, C_t), G_t.transpose()) + W_t)[0,0]
                #print(R_ti)
            #print(a_t.shape)
            #print(s_t.shape)
            sh_t = m_t + np.dot(np.dot(np.dot(C_t, G_t.transpose() ),R_ti ),(s_t - a_t) )
            H_t = C_t - np.dot(np.dot(np.dot(np.dot(C_t,G_t.transpose() ),R_ti ),G_t ),C_t )
            #print("ht",H_t)
    return st





def shared_array(shape):
    """
    Form a shared memory numpy array.

    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """
    shared_array_base = Array(ctypes.c_double, shape[0] * shape[1])
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return shared_array

def divide_work(num,coren):
    part = []
    rest = h % coren
    o = int((h - rest)/coren)
    temp = 0
    for i in range(coren):
        if i == (coren-1):
            temp = rest
        part.append(range(i*o,i * o + o + temp ) )
    return part


# Form a shared array and a lock, to protect access to shared memory.

k = demand[1].shape[1]
kp = k - 1

#h = len(demand)
h = 387

psia = np.array([0.] * h * k).reshape((h,k))
psid = {}
glist = list()
start = 0
ind = np.array([0]*h*2).reshape((h,2))

for i in range(h):
    psid[i] = np.array([0.]*demand[i].shape[0]*k).reshape((demand[i].shape[0],k))
    ind[i,0] = start
    ind[i,1] = start + demand[i].shape[0] - 1
    glist.append(range(start,start+demand[i].shape[0]))
    start = start+ demand[i].shape[0]
#print(glist)
#print(ind)
#psi = shared_array((t,k))
psi = np.concatenate([psid[x] for x in sorted(psid)], 0)
t = psi.shape[0]

if len(sys.argv) > 2 :
     c = int(sys.argv[2])
else: c = 2

ran = divide_work(h,c)

### 2. MCMC settings
burnin =500
mcmc = 1000
#if len(sys.argv) > 2 :
#    if len(sys.argv) > 3:
#        mcmc = sys.argv[2]
#    burnin=sys.argv[3]


thin = 1
nmcmc = burnin + thin*mcmc

### 3.  Prior Setting





### 4. Initial Value



#psi[:,:] = psi2

zdata = np.array([1.] * h).reshape((h, 1))
#rw =multivariate_normal.rvs(mean=0.,cov= 1.,size=t)
#rw2 =multivariate_normal.rvs(mean=0.,cov= 1.,size=t)
#zdata2 = np.array([1.] * t).reshape((t, 1))
#zdata3 = np.array([1.] * t * 2).reshape((t, 2))
#beta = np.array([0.] * h * 2).reshape((h, 2))
#count = 0

rankz = zdata.shape[1]

#####################################

gu0 = np.zeros((k,1))
for i0 in range(1):
    gu0[i0,i0] = 1
gv0=np.diag([100.]*1)
gv0i = np.linalg.inv(gv0)
ss0=2
sr0=2
srn=sr0+t

bu0 = np.zeros((k,2))
for i in range(2):
    bu0[i,i] = 1
bv0 = np.diag([100.]*2)
bv0i = np.linalg.inv(bv0)

ds0=2
dr0=2
drn = dr0 + t


pu0 = np.zeros((1,1))
pv0 = np.diag([10.]*1)

pv0i = np.linalg.inv(pv0)
pv0iu0 = pv0i.dot(pu0)

##############vb
f0 = 1+2
f0n = f0 + h
g0 = np.diag([f0])

tpa = np.diag([0.1]*k)
tpp = np.diag([0.1]*kp)

gama = np.zeros((t,k))
gamm = np.zeros((t,k))
gams = gamm
gamg = np.zeros((mcmc,k))

gh = gu0
ghg = np.zeros((mcmc,k))

fa = np.zeros((t,1))
fm = np.zeros((t,1))
fs = np.zeros((t,1))
fg = np.zeros((mcmc,t*1))

sigma = np.diag([1.]*k)
sigmai = np.linalg.inv(sigma)
sigmag = np.zeros((mcmc,k))

vf = np.diag([1.])
vfi = np.linalg.inv(vf)
vfg = np.zeros((mcmc,1))

### baseline
psia = np.zeros((t,k))
psim = np.zeros((t,k))
psis = psim
psig = np.zeros((mcmc,k))

bh = bu0
bhg = np.zeros((mcmc,k*2))

ga = np.zeros((t,2))
gm = ga
gs = ga
gg = np.zeros((mcmc,t*2))

vg = np.diag([1.,1.])

delta = np.diag([1.]*kp)
deltai = np.linalg.inv(delta)
deltag = np.zeros((mcmc,kp))

betaa = np.zeros((h,2))
betam = betaa
betas = betam

phi = np.zeros((1,2))
phig = np.zeros((mcmc,2))

vb = np.diag([1.]*2)
vbi = np.linalg.inv(vb)
vbg = np.zeros((mcmc,4))

cmn = k + kp
kl = 1+2

m0 = np.zeros((kl,1))
c0 = np.diag([100.]*kl)
print("c0",c0)
abh = np.zeros((cmn,kl))
sdm = np.diag([1.] * cmn)

kt = np.array([1.]*t).reshape((t,1))
ktm = np.zeros((t,1))
kts = kt

r = np.zeros((h,1))
rg = np.zeros((mcmc,h))

lla = np.zeros((t,1))
llg = np.zeros((mcmc,t))

dems = demand[0]
pris = prices[0]

mis = None


for i in range(1,h):
    dems = np.vstack((dems,demand[i]))
    pris = np.vstack((pris,prices[i]))

dems = np.array(dems).reshape((t,k))
pris = np.array(pris).reshape((t,k))


print(ran)
for i in range(len(ran)):
    for j in ran[i]:
        for jj in glist[j]:
            lla[jj] = mdcev(psia[jj, :].transpose(), gama[jj, :].transpose(), dems[jj, :],
                           pris[jj, :])
#for i in range(t):
#    lla[i] =  mdcev(psia[i, :].transpose(), gama[i, :],dems[i,:],pris[i,:] )
art = np.zeros((t,2))
print(lla)

#for i in range(h):
#    parh = {"psi":psi[glist[i],:],"gam":gama[glist[i], :]}
#    lla[i] = mdcev(parh, demand[i], prices[i])
#print(lla)

itea = np.array([0] * h * 2).reshape((h, 2))

### parallel
#def cal(i,psia,gama,lla,vp,vg):
#def cal(ran):
def cal(psi,gama,lla,fa,ga,kt,r,betaa,zphi,bh,gh,abh,ran):
    itea = np.zeros((t,2))
    for i in ran:
        #print("now in:",i)
        #parh = {"psi": psi[glist[i],:], "gam": gama[glist[i], :]}
        #priorh = {"pu0": zt[glist[i], :], "pv0": vp, "gu0": zp[glist[i], :], "gv0": vg,
        #          "tpp": tpp[i] * np.diag(np.array([1] * kp)),
        #          "tpg": tpg[i] * np.diag(np.array([1] * k)),"f":zdata2[glist[i], :],"g":zdata3[glist[i], :]}
        #print(lla[glist[i]])
        outh = rwmh(demand[i], prices[i],1,lla[glist[i]],gama[glist[i],:],fa[glist[i],:],
                    psi[glist[i],:],ga[glist[i],:],kt[glist[i]],r[i],betaa[i,:].transpose(),zphi[i,:].transpose(),bh,gh,abh)
        #if i == 1:
        #     print("psia", psia[i, :])
        psi[glist[i],:] = outh["psi"]
        gama[glist[i], :] = outh["gam"]
        fa[glist[i],:] = outh["f"]
        ga[glist[i],:] = outh["g"]
        kt[glist[i]] = outh["kt"].reshape((len(outh["kt"]),1))
        betaa[i,:] = outh["betai"]
        itea[glist[i]] = outh["ite"]
        lla[glist[i]] = outh["ll"]
    #print("complete!")
    t1 =list(glist[list(ran)[0]])[0]
    t2 = list(glist[list(ran)[-1]])[-1]
    output = {"psia":psi[t1:t2,:] ,"gama":gama[t1:t2,:],"lla":lla[t1:t2],"itea":itea[t1:t2],
              "fa":fa[t1:t2,:],"ga":ga[t1:t2,:],"kt":kt[t1:t2],"betaa":betaa[ran,:]}
    return output


v = np.diag([0.1] * k)
iv = np.linalg.inv(v)

#def hmcmc():




if __name__ == '__main__':
    if(len(sys.argv) > 1) :
        coren = sys.argv[1]
    argList = range(h)
    coren = 4
    p = Pool(coren)
    for imcmc in range(nmcmc):
        print(imcmc)
        try:
            ff =open("process.txt","w+")
            ff.write(imcmc)
        except:
            pass

        start = time.time()
        #print("psia",psia[1,:])
        #if(imcmc >0 ):print("before psia2",psia2)
        #zt = np.dot(zdata2, theta)
        #zp =  np.dot(zdata3, phi)
        itea = np.zeros((t,2))
        zphi = zdata.dot(phi)
        cal2 = partial(cal,psi,gama,lla,fa,ga,kt,r,betaa,zphi,bh,gh,abh)
        result = p.map(cal2,ran)
        #result = p.map(cal2,argList2)

        for i in range(len(ran)):
            t1 = list(glist[list(ran[i])[0]])[0]
            t2 = list(glist[list(ran[i])[-1]])[-1]
            #print("t1:t2",t1," ",t2)
            #print(ran[i])
            #print(list(glist[list(ran[i])[0]]))
            psi[t1:t2,:]= result[i]["psia"]
            gama[t1:t2,:] = result[i]["gama"]
            lla[t1:t2]    = result[i]["lla"]
            itea[t1:t2,:] = result[i]["itea"]
            fa[t1:t2,:] = result[i]["fa"]
            ga[t1:t2,:] = result[i]["ga"]
            betaa[ran[i],:] = result[i]["betaa"]
            kt[t1:t2,:] = result[i]["kt"]
            #print("over")
        art  += itea
        Eresult = EFAs(gama,fa,psia,ga)

        ## ah & bh
        gh = Eresult["gh"]
        bh = Eresult["bh"]
        abh[0:(k),0] = gh.reshape((k))
        abh[k:(k+kp),1:(1+2)] = bh[0:(kp),:]

        ## phi & vb
        #out01 = rmultireg(betaa[:,j],zdata,phi,vb[j,j],pu0,pv0)
        #out1 = rmultireg(y=psi[:, 0:kp], x=zdata2, bbar=tu0, a=tv0, nu=pf0, v=pg0, n=1)
        for j in range(2):
            #print(betaa[:,j].shape)
            out01 = bmreg(betaa[:,j].reshape((h,1)),zdata,phi,vb[j,j],pu0,pv0,f0,g0)
            phi[0,j] = out01["s_theta"]
            #vb[j,j] = out01["s_lambda"]
        if imcmc >= burnin :
            jmcmc = int((imcmc - burnin)/thin)
            #print(psi[:,1])

            psig[jmcmc,:] = np.mean(psi, axis=0)
            gamg[jmcmc,:] = np.mean(gama, axis=0)
            gamm += gama
            gams += gama ** 2

            fm += fa
            fs += fa ** 2
            fg[jmcmc,:] = fa.reshape((-1))

            ghg[jmcmc,:] = gh.reshape((-1))
            sigmag[jmcmc,:] = np.diag(sigma)

            ## baseline
            psim += psi
            psis += psi ** 2

            gm += ga
            gs += ga ** 2
            gg[jmcmc,:] = ga.reshape((-1))

            bhg[jmcmc,:] = bh.reshape((-1))
            deltag[jmcmc,:] = np.diag(delta)

            betam += betaa
            betas += betaa ** 2
            print(np.mean(psi,axis=0))
            print(np.mean(gama,axis=0))
            print("beta",betaa[0:10,:])
            #print("delta",delta)

            phig[jmcmc,:] = phi.reshape((-1))
            vbg[jmcmc,:] = vb.reshape((-1))
            print(phi)

            ktm += kt
            llg[jmcmc,:] =  lla.transpose()
            if jmcmc % 100 == 0:
                np.savez("D:/OneDrive/dat/resultn.npz", psig=psig,
                         gamg=gamg, psim=psim, psis=psis, ktm=ktm, fm=fm, fs=fs, fg=fg,
                         ghg=ghg, gm=gm, gs=gs, gg=gg, bhg=bhg, deltag=deltag,
                         phig=phig, llg=llg, sigmag=sigmag,
                         vbg=vbg, gamm=gamm, gams=gams, mcmc=mcmc,
                         betam=betam, betas=betas, glist=glist,dems=dems,pris=pris,ran=ran,ind=ind)

        print("spend time:",time.time()-start)

    #output = {"psig":psig,"gamg":gamg,"thetag":thetag,
    #          "phig":phig,"llg":llg,"vpg":vpg,"vgg":vgg,
    #          "beta1":betag1,"beta2":betag2,"f":zdata2,"g":zdata3}
    if len(sys.argv) > 1:
        if sys.argv[1] == 1:
            np.savez("/stfs1/uhome/y90027/MCMC/result.npz", psig=psig, thetam=thetam,thetas=thetas,
                     gamg=gamg, psim=psim, psis=psis, ktm = ktm,fm=fm,fs=fs,fg=fg,
                     ghg = ghg, gm = gm , gs = gs , gg = gg, bhg = bhg,deltag = deltag,
                     phig = phig,llg = llg, sigmag = sigmag,
                     vbg=vbg, gamm=gamm, gams=gams, phim=phim, phis=phis,
                     betam = betam,betas=betas,glist=glist)
    else:
        np.savez("D:/OneDrive/dat/resultn.npz", psig=psig,
                     gamg=gamg, psim=psim, psis=psis, ktm = ktm,fm=fm,fs=fs,fg=fg,
                     ghg = ghg, gm = gm , gs = gs , gg = gg, bhg = bhg,deltag = deltag,
                     phig = phig,llg = llg, sigmag = sigmag,
                     vbg=vbg, gamm=gamm, gams=gams,mcmc=mcmc,
                     betam = betam,betas=betas,glist=glist,dems=dems,pris=pris,ran=ran,ind=ind)
    '''
    y = np.array([0,1,4,7,9,6,2,3]).reshape((4,2))
    x = np.array([2,3,4,8,2,8,9,2]).reshape((4,2))


    res = rmultireg(y=psia[:,0:kp],x=zdata,bbar=tu0,a=tv0,nu=pf0,v=pg0,n=1)
    print("B:",res["B"])
    print("Sigma:", res["Sigma"][0])
    print( xpnd(np.array([1,2,3,4,5,6]) ) )
    print(rwishart(nu=pf0,v=pg0))
    '''

    ## MCMC
    outp =hmcmc()

    ## Posterior Mean
    ## Posterior s.d.
    psim = np.mean(outp["psig"],axis=0).reshape((t,k))
    psis = np.array([0.]*t*k)
    for i in range(int(t*k)):
        psis[i] = np.std(outp["psig"][:,i])

    #psis = np.std(outp["psig"],axis=0).reshape((t,k))
    psis = psis.reshape((t,k))
    thetam = np.mean(outp["thetag"],axis=0).reshape((rankz,kp))
    #thetas= np.array([0.]*t*k)
    #for i in range(int(kp)):
    #    thetas[i] = np.std(outp["psig"][:,i])

    thetas = np.std(outp["thetag"], axis=0).reshape((rankz, kp))
    thetam[0,0] = 1
    thetas[0,0] = 0
    print("theta")
    print(thetam)
    print("f")
    print(outp["f"])
    print("g")
    print(outp["g"])

    vpm = xpnd (  np.mean(outp["vpg"],axis=0) )
    vps= xpnd(np.std(outp["vpg"], axis=0))


    gamm = np.mean(outp["gamg"], axis=0).reshape((t, k))
    gams = np.std(outp["gamg"], axis=0).reshape((t, k))



    phim = np.mean(outp["phig"],axis=0).reshape((2,k))
    phis = np.std(outp["phig"],axis=0).reshape((2,k))
    phim[0,0] = phim[1,1] = 1
    phim[1,0] = 0
    phis[0,0] = phis[0,1] = phis[1,1] = 0

    print("phi")
    print(phim)

    vgm = xpnd( np.mean(outp["vgg"],axis=0) )
    vgs = xpnd(np.std(outp["vgg"], axis=0))

    betag = np.array([0.]*h*2).reshape((h,2))
    betag[:,0] = np.mean(outp["beta1"],axis=0)
    betag[:,1] = np.mean(outp["beta2"],axis=0)


    accept_rate = mcmc/itea
    plt.figure(1)
    subn = 210
    plt.figure(1)
    count = 0
    for i in range(90):
        plt.plot(outp["psig"][:, i])
        title = "psia" + str(i)
        plt.title(title)
        plt.savefig("d:/onedrive/pic/psia/" + title + ".png")
        plt.clf()
    for i in range(90):
        plt.plot(outp["gamg"][:, i])
        title = "gama" + str(i)
        plt.title(title)
        plt.savefig("d:/onedrive/pic/gama/" + title + ".png")
        plt.clf()
    ## dic
    llm =np.zeros(h)
    for i in range(h):
        parm = {"psi":psim[glist[i],:],"gam":gamm[glist[i],:]}
        llm[i] = mdcev(parm,demand[i],prices[i])
    print("llm",llm)
    lml = 1/( np.sum(1/llm)/h)
    try:
        lml2 = np.log(1 / ( np.sum(1 / np.exp(llm) )/h))
    except:
        pass
    dic = -4*np.sum(np.mean(llg,axis=0)) + 2*np.sum(llm)
    ### print
    #print("llm:",llm)
    #print("psim:",psim)
    #print("psis",psis)
    print("lml",lml)
    print("lml2",lml2)
    print("dic",dic)

    print("psim")
    print(psim)
    print("psis")
    print(psis)
    print("beta")
    print(betag)
    '''
    np.savez("/stfs1/uhome/y90027/MCMC/result.npz",psig=outp["psig"],thetag=outp["thetag"],vpg=outp["thetag"],
             gamg=outp["gamg"],phig=outp["phig"],vgg=outp["vgg"],
             dic=dic,psim=psim,psis=psis,thetam=thetam,
             vpm=vpm, vps=vps, gamm=gamm, gams=gams, phim=phim, phis=phis, vgm=vgm, vgs=vgs,accept_rate=accept_rate,lml=lml)
    '''






            # for i in range(90):
    #     plt.plot(outp["psig"][:, i])
    #     title = "psia" + str(i)
    #     plt.title(title)
    #     plt.savefig("/stfs1/uhome/y90027/MCMC/PIC/psia/" + title + ".png")
    #     plt.clf()
    # for i in range(90):
    #     plt.plot(outp["gamg"][:, i])
    #     title = "gama" + str(i)
    #     plt.title(title)
    #     plt.savefig("/stfs1/uhome/y90027/MCMC/PIC/gama/" + title + ".png")
    #     plt.clf()




