import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker


import scipy
from scipy import interpolate, special
from  scipy.special import j1
from scipy.interpolate import interpn, interp1d, interp2d, RegularGridInterpolator
import struct
import camb
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw, concentration

cosmology.setCosmology('WMAP7')

sys.path.append("/home/chenzy/code/mypycode/")
import fftanalysis as fa

Boxlen = 1200 #Mpc/h
ns=0.968
sigma_8 = 0.83
h=0.71
Om0=0.268
Ob0=0.0445

pm=2.775*10**11*Om0*Boxlen**3/3072.**3 #M_sun h^2 /Mpc^-3
c_speed=2.9979*10**5#km/s
ne0=9.83*Ob0*h**2	#m^-3
sigma_t=6.65246*10**(-29)	#m^2
Mpc2m=3.0856*10**22  
T_cmb=2.728*10**6

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.268,Ob0=0.0445,  Tcmb0=2.725)

cmbsur=[[11,1.26,"ACT"],[0.78*60,5,"PLANCK"]]

#======================   General      ======================
class Snap_Info:
    def __init__(self):
        #self.zstep = np.array([5000,4724,4338,4098,3982,3656,3356,3080,2826,2593,2448,2378,2181,2000,1942,1780,1631,1494,1328,1216])
        #self.vel_norm = np.array([17400000.0, 15923808.3, 14020373.5, 12931188.4, 12430474.5, 11111965.1, 10011775.9, 9091734.5, 8318855.9, 7667665.8, 7288078.4, 7111356.2, 6634895.0, 6221216.6, 6092865.1, 5743409.9, 5431402.9, 5149821.4, 4811557.1, 4582649.9])
        #self.redshift = np.array([0., 0.058, 0.151, 0.218, 0.253, 0.364, 0.485, 0.616, 0.76, 0.916, 1.028, 1.087, 1.272, 1.474, 1.547, 1.774, 2.023, 2.293, 2.695, 3.025])
        
        self.zstep = np.array([5000., 4860., 4724., 4591., 4463., 4338., 4216., 4098., 3982.,3871., 3762., 3656., 3553., 3453., 3356., 3261., 3169., 3080.,2993., 2908., 2826., 2746., 2668., 2593., 2519., 2448., 2378.,2310., 2245., 2181., 2119., 2058., 2000., 1942., 1887., 1833.,1780., 1729., 1679., 1631., 1584., 1539., 1494., 1451., 1409., 1368., 1328., 1290., 1252., 1216., 1180., 1146., 1112., 1080., 1048., 1017.,  987.,  958.,  929.,  902.,  875.,  849.,  824., 799.,  775.,  752.,  729.,  707.,  685.,  665.,  644.,  625., 605.,  587.,  569.,  551.,  534.,  517.,  501.,  485.,  470., 455.,  441.,  427.,  413.,  399.,  387.,  374.,  362.,  350., 338.,  327.,  316.,  305.,  295.,  285.,  275.,  266.,  256., 247.])
        self.redshift = np.array([ 0.   ,  0.029,  0.058,  0.088,  0.119,  0.151,  0.184,  0.218, 0.253,  0.289,  0.326,  0.364,  0.403,  0.444,  0.485,  0.528, 0.572,  0.616,  0.663,  0.711,  0.76 ,  0.811,  0.863,  0.916, 0.972,  1.028,  1.087,  1.147,  1.208,  1.272,  1.338,  1.406, 1.474,  1.547,  1.62 ,  1.696,  1.774,  1.855,  1.938,  2.023, 2.11 ,  2.199,  2.293,  2.389,  2.487,  2.589,  2.695,  2.801, 2.913,  3.025,  3.145,  3.264,  3.391,  3.517,  3.65 ,  3.787, 3.928,  4.072,  4.224,  4.375,  4.534,  4.697,  4.863,  5.039, 5.218,  5.4  ,  5.592,  5.788,  5.995,  6.195,  6.418,  6.632, 6.87 ,  7.098,  7.339,  7.596,  7.853,  8.125,  8.398,  8.687,  8.975,  9.281,  9.583,  9.904, 10.245, 10.608, 10.938, 11.318, 11.691, 12.087, 12.508, 12.919, 13.355, 13.82 , 14.27 , 14.747, 15.256, 15.742, 16.318, 16.871])
        self.vel_norm = np.array([17400000.0, 16639191.9, 15923808.3, 15246772.7, 14616239.4, 14020373.5, 13457708.4, 12931188.4, 12430474.5, 11966922.7, 11526445.3, 11111965.1, 10722191.7, 10355868.7, 10011775.9, 9685350.0, 9379040.3, 9091734.5, 8819281.0, 8560928.6, 8318855.9, 8089283.8, 7871541.4, 7667665.8, 7471605.7, 7288078.4, 7111356.2, 6943523.4, 6786469.7, 6634895.0, 6490800.8, 6351513.0, 6221216.6, 6092865.1, 5972816.8, 5856399.8, 5743409.9, 5635755.0, 5531117.3, 5431402.9, 5334364.2, 5241914.8, 5149821.4, 5062065.1, 4976495.2, 4893020.1, 4811557.1, 4734073.3, 4656426.5, 4582649.9, 4508596.7, 4438342.8, 4367722.7, 4300866.4, 4233576.4, 4167922.1, 4103896.4, 4041498.5, 3978553.1, 3919411.4, 3859706.6, 3801637.5, 3745228.2, 3688214.8, 3632873.3, 3579239.9, 3524982.7, 3472463.5, 3419298.9, 3370373.7, 3318357.2, 3270692.9, 3219866.7, 3173518.7, 3126566.7, 3078978.0, 3033417.5, 2987226.9, 2943147.9, 2898451.0, 2855958.0, 2812865.3, 2772077.5, 2730714.2, 2688746.8, 2646144.6, 2609098.1, 2568384.8, 2530242.4, 2491535.0, 2452234.7, 2415662.9, 2378542.6, 2340846.9, 2306054.3, 2270738.6, 2234874.3, 2202104.9, 2165121.0, 2131293.8])/c_speed
        

    def get_redshift(self, step):
        label  = np.where(self.zstep == step)
        return self.redshift[label]
    
    def get_vel_norm(self, step):
        label  = np.where(self.zstep == step)
        return self.vel_norm[label]
snap_info = Snap_Info()
 
#======================   Cl: CMB, noise, kSZ      ======================   
def cmbcl(H0=h*100, omch2=(Om0-Ob0)*h**2, ombh2=Ob0*h**2, mnu=0.06, omk=0, tau=0.06,As=2e-9, ns=ns, r=0,lmax=10000):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax+1, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars,lmax=lmax,raw_cl=True)
    l=np.arange(len(powers["total"][:,0]))
    np.savez("/home/chenzy/code/kSZ_forecast/data/cllensedcmb.npz",l=l,cl=powers["total"][:,0])
    return l, powers["total"][:,0]
def cmbnoise(T_cmb=2.728*10**6,Delta_T=0.78*60,sigma=5,lmax=10000):
    l=np.arange(lmax+1)
    arcmin2rad=1.0/60/180*np.pi
    cln=(Delta_T*arcmin2rad/T_cmb)**2*np.exp(l*(l+1)*(sigma*arcmin2rad)**2/8/np.log(2))
    return l,cln
def cal_cl_kSZ(z_min=0, z_max=5, n_zbin=501, lgl_min=2, lgl_max=4, n_lglbin=80, grid=1024):
    l_array = 10**np.linspace(lgl_min, lgl_max, n_lglbin)
    redshift_inte_array = np.linspace(z_min, z_max, n_zbin)
    d_coming_array = cosmo.comoving_distance(redshift_inte_array).value  #Mpc
    dd_coming_array = (d_coming_array[1:]-d_coming_array[:-1])
    d_coming_array = (d_coming_array[1:]+d_coming_array[:-1])/2
    redshift_inte_array = (redshift_inte_array[1:]+redshift_inte_array[:-1])/2
    
    Delta2_B = []
    redshift = []
    #interp of Delta2_B
    for Snapshot in [5000, 3356, 2448, 1631, 1216, 958, 799]:
        file_name = "/home/chenzy/code/kSZ_forecast/data/Power_Spectrum_momp_B_"+str(Snapshot)+"_"+str(grid)+".npz"
        f = np.load(file_name)
        
        k = f["k"]
        Delta2_B.append(f["Delta2_B"])
        redshift.append(f["redshift"])
        
    f_Delta2_B = interp2d(k, np.array(redshift),  np.array(Delta2_B), fill_value = 0)
    
    Delta2_B = np.zeros((len(redshift_inte_array), len(l_array)))
    for i in range(len(redshift_inte_array)):
        Delta2_B[i,:] = f_Delta2_B(l_array/d_coming_array[i]/h, redshift_inte_array[i])
        
    cnorm = 16*np.pi**2*(ne0*sigma_t)**2/2*Mpc2m**2
    cnorm_z = (1+redshift_inte_array)**4*d_coming_array*dd_coming_array
    cnorm_l = 1/(2*l_array+1)**3
    
    #print(cnorm_z.shape, cnorm_l.shape)

    Cl_kSZ = cnorm*np.sum(cnorm_z.reshape(-1,1)*cnorm_l.reshape(1,-1)*Delta2_B, axis=0)
    
    return l_array, Cl_kSZ
def cl2map2d(l,cl,ang,grid):
	# unit of ang is rad, is the length of the square
	clintp=interpolate.interp1d(l,cl)

	#gen a noise map, pk=1 white noise
	np.random.seed()
	map_noise=np.random.rand(grid,grid)
	coef1=2*np.pi*map_noise[0:grid:2,:]
	coef2=np.sqrt(-2*np.log(map_noise[1:grid:2,:]))
	map_noise[0:grid:2,:]=coef2*np.cos(coef1)
	map_noise[1:grid:2,:]=coef2*np.sin(coef1)

	map_noisek =np.fft.rfft2(map_noise)
	#coef of map_noisek from cl
	llen=np.zeros((grid,grid//2+1))
	for i in range(grid):
		if i>grid/2: ii=i-grid
		else: ii=i
		for j in range(grid//2+1):
			jj=j
			llen[i,j]=np.sqrt(ii**2+jj**2)*2*np.pi/ang

	map_noisek*=np.sqrt(clintp(llen)*(grid/ang)**2)
	map2d=np.fft.irfft2(map_noisek)
	return map2d
def cmb2d(thelen,grid,Delta_T=11,sigma=1.26):
    ll,cl_cmb=cmbcl()
    map2d_cmb=cl2map2d(ll,cl_cmb,thelen,grid)
    l,cl_noise=cmbnoise(Delta_T=Delta_T,sigma=sigma)
    map2d_noise=cl2map2d(l,cl_noise,thelen,grid)
    map2d=map2d_cmb+map2d_noise
    '''
    ps,l=fa.CalPS2d(map2d,grid,mapscale=(thelen/grid**2)**2,kscale=2*np.pi/thelen)
    plt.plot(l,ps*l**2/2/np.pi)
    plt.plot(ll,cl_cmb*ll**2/2/np.pi)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(10**-14,10**-9)
    plt.grid()
    #plt.savefig("../../pic/a.png")
    '''
    return map2d

#======================   reconstruction      ======================
def vel2theta(vel, boxlen):
    grid=vel.shape[0]
    kmod,kx,ky,kz=fa.karray(grid,True)
    kmod[0,0,0]=1
    velkx=np.fft.rfftn(vel[:,:,:,0])
    velky=np.fft.rfftn(vel[:,:,:,1])
    velkz=np.fft.rfftn(vel[:,:,:,2])

    thetak=(kx*velkx+ky*velky+kz*velkz)*(1j)*(2*np.pi/boxlen)

    theta=np.fft.irfftn(thetak)
    return theta
def theta2vel(theta, boxlen):
    grid=theta.shape[0]
    kmod,kx,ky,kz=fa.karray(grid,True)
    kmod[0,0,0]=1
    vel=np.zeros((grid,grid,grid,3))
    thetak=np.fft.rfftn(theta)
    vel[:,:,:,0]=np.fft.irfftn(thetak*kx/kmod**2*(-1j))/(2*np.pi/boxlen)
    vel[:,:,:,1]=np.fft.irfftn(thetak*ky/kmod**2*(-1j))/(2*np.pi/boxlen)
    vel[:,:,:,2]=np.fft.irfftn(thetak*kz/kmod**2*(-1j))/(2*np.pi/boxlen)

    return vel
def momden2vel(den, mom):
    #den,mom -> vel, and transfer mom=(1+d)v -> dv
    #t0=time.time()
    grid = den.shape[0]
    vel=np.zeros(mom.shape)
    dentemp=np.ones(den.shape)
    dentemp[np.where(den>0)]=den[np.where(den>0)]
    vel=mom/dentemp.reshape(grid,grid,grid,1)
    #mom=vel*(den-1).reshape(grid,grid,grid,1)
    #del dentemp,denp
    #print("den/mom convert to vel  "+str(np.round(time.time()-t0,3))+'s')
    return vel
def den2theta(den, redshift, cosmo_model, grow_rate_index=4/7.):
    grid=den.shape[0]

    H = cosmo.H(z=redshift).value #km/s/(Mpc/h)
    f = cosmo.Om(z=redshift)**grow_rate_index

    return -H*f*den
def den_filter_g(den, Rg, Boxlen):
    grid=den.shape[0]
    denk=np.fft.rfftn(den)
    kmod=fa.karray(grid)
    denk_f = denk*np.exp(-(kmod*Rg*2*np.pi/Boxlen)**2/2)
    den_f = np.fft.irfftn(denk_f)
    
    return den_f
def wiener_filter_theta_esti(denh, thetap, boxlen, nkbin = 30):
	thetak=np.fft.rfftn(thetap)
	denhk=np.fft.rfftn(denh)
	grid=denh.shape[0]

	kmod,kx,ky,kz=fa.karray(grid,True)
	kx[0,:,:]=1
	ky[:,0,:]=1
	kz[:,:,0]=1
	kcoor=np.zeros((len(kx.flatten()),3))
	kcoor[:,0]=np.log10(np.abs(kx).flatten())
	kcoor[:,1]=np.log10(np.abs(ky).flatten())
	kcoor[:,2]=np.log10(np.abs(kz).flatten())
	cor_the_delta_f=np.histogramdd(kcoor,bins=nkbin,weights=np.real(thetak.flatten()*np.conjugate(denhk).flatten()))
	ps_delta=np.histogramdd(kcoor,bins=nkbin,weights=np.real(denhk.flatten()*np.conjugate(denhk).flatten()))
	W=cor_the_delta_f[0]/ps_delta[0]
	kx=10**ps_delta[1][0]*2*np.pi/boxlen
	ky=10**ps_delta[1][1]*2*np.pi/boxlen
	kz=10**ps_delta[1][2]*2*np.pi/boxlen
	return W, kx,ky,kz
def wiener_filter_den2theta(den,W,kx,ky,kz, boxlen):
	grid=den.shape[0]
	kmod,kxx,kyy,kzz=fa.karray(grid,True)
	kx=(kx[1:]+kx[:-1])/2
	ky=(ky[1:]+ky[:-1])/2
	kz=(kz[1:]+kz[:-1])/2
	kx=kx[6:];ky=ky[6:];kz=kz[6:]

	kx[0]=0;kx[-1]=6
	ky[0]=0;ky[-1]=6
	kz[0]=0;kz[-1]=6

	W_new=np.zeros((24,24,24))
	W_new[0,0,0]=(np.mean(W[0,0,7:])+np.mean(W[0,7:,0])+np.mean(W[7:,0,0]))/3
	W_new[0,0,1:]=W[0,0,7:]
	W_new[0,1:,0]=W[0,7:,0]
	W_new[1:,0,0]=W[7:,0,0]
	W_new[0,1:,1:]=W[0,7:,7:]
	W_new[1:,0,1:]=W[7:,0,7:]
	W_new[1:,1:,0]=W[0,7:,7:]
	W_new[1:,1:,1:]=W[7:,7:,7:]
	W=W_new

	kcoor=np.zeros((len(kxx.flatten()),3))
	kcoor[:,0]=np.abs(kxx).flatten()*2*np.pi/boxlen
	kcoor[:,1]=np.abs(kyy).flatten()*2*np.pi/boxlen
	kcoor[:,2]=np.abs(kzz).flatten()*2*np.pi/boxlen
	wintp=interpn(np.array([kx,ky,kz]),W,kcoor)
	wintp=wintp.reshape(grid,grid,grid//2+1)

	thetak=np.fft.rfftn(den)*wintp
	return np.fft.irfftn(thetak)
def wiener_filter_den(f ,bg,pk,k,kscale,n,grid):
#pk [Mpc/h]**3
#k [h/Mpc]
#w=pk/(pk+1/n)
	kmod,kx,ky,kz=fa.karray(grid,True)

	#wiener filter
	#pkden,k=fa.CalPS(f,grid,mapscale=(boxlen/grid**2)**3)
	W=bg**2*pk/(bg**2*pk+1/n)
	kinp=np.zeros(len(k)+2);kinp[-1]=500;kinp[1:-1]=k
	Winp=np.zeros(len(W)+2);Winp[-1]=W[-1];Winp[0]=1;Winp[1:-1]=W
	W_inp=interpolate.interp1d(kinp,Winp)
	W_arr=W_inp(kmod*kscale)
	fk=np.fft.rfftn(f)
	fk*=np.sqrt(W_arr)
	ff=np.fft.irfftn(fk)
	return ff

#======================   halos      ======================
def ReadHalos(zstep):
    name='/home/cossim/CosmicGrowth/6620/fof/gcatdbb02.6620.'+str(zstep);
    print("Reading "+name)
    print("Redshift = ",str(snap_info.get_redshift(zstep)))
    pos=open(name,'rb')
    pos.read(4)
    ng=struct.unpack('l',pos.read(8))[0]	#number of haloes
    print("The number of halo"+str(ng))
    nmin=pos.read(8);nmin=struct.unpack('l',nmin)[0]
    print('ng:',ng,'nmin:',nmin)
    pos.read(8)
    nrich=pos.read(8*ng);nrich=np.array(struct.unpack(str(ng)+'l',nrich))
    pos.read(8)
    halox=np.zeros((ng,3))
    halov=np.zeros((ng,3))
    haloxx=pos.read(4*ng);halox[:,0]=np.array(struct.unpack(str(ng)+'f',haloxx));pos.read(8)
    haloxy=pos.read(4*ng);halox[:,1]=np.array(struct.unpack(str(ng)+'f',haloxy));pos.read(8)
    haloxz=pos.read(4*ng);halox[:,2]=np.array(struct.unpack(str(ng)+'f',haloxz));pos.read(8)
    halovx=pos.read(4*ng);halov[:,0]=np.array(struct.unpack(str(ng)+'f',halovx));pos.read(8)
    halovy=pos.read(4*ng);halov[:,1]=np.array(struct.unpack(str(ng)+'f',halovy));pos.read(8)
    halovz=pos.read(4*ng);halov[:,2]=np.array(struct.unpack(str(ng)+'f',halovz));
    pos.close()
    return halox,halov,nrich,ng
def get_halo_sample_all(snapshot):
#return halox,halov
    halox,halov,mh,nh=ReadHalos(snapshot)
    return halox,halov, mh
def save_halo_np(snapshot):
    halox,halov, mh = get_halo_sample_all(snapshot)
    np.savez("/home/chenzy/code/kSZ_forecast/halo_Temporary_storage/halos_"+str(snapshot), halox=halox, halov=halov, mh=mh)
def ReadHalos_np(zstep):
    f = np.load("/home/chenzy/code/kSZ_forecast/halo_Temporary_storage/halos_"+str(zstep)+".npz")
    halox = f["halox"]
    halov = f["halov"]
    mh = f["mh"]
    return  halox,halov, mh
def den_NGP(x,grid):
#return den
    den=np.histogramdd(x,bins=(grid,grid,grid))[0]
    den=den*1.0/np.sum(den)*grid**3
    return den
def den_CIC(x,grid,boxlen=1):
#return den
    x=x*grid*1.0/boxlen
    xdec=x-np.floor(x)
    den=np.zeros((grid,grid,grid))
    for xx in [0,1]:
        for yy in [0,1]:
            for zz in [0,1]:
                w=np.abs(-xdec[:,0]+1-xx)*np.abs(-xdec[:,1]+1-yy)*np.abs(-xdec[:,2]+1-zz)
                xi=np.zeros(x.shape)
                xi[:,0]=(x[:,0]+xx)%grid
                xi[:,1]=(x[:,1]+yy)%grid
                xi[:,2]=(x[:,2]+zz)%grid
                den+=np.histogramdd(xi,bins=(grid,grid,grid),weights=w)[0]
    den=den*1.0/np.sum(den)*grid**3
    return den
def den_TSC(x,grid,boxlen=1):
#return den
    cat=ArrayCatalog({'Position' : x})
    den_mesh=cat.to_mesh(Nmesh=grid, BoxSize=boxlen, resampler="TSC", compensated=True, interlaced=False)
    return den_mesh.preview()
def add_RSD(x,v,n,H):
    dx=np.dot(v,n)/H		# comoving  Mpc/h
    #x+=dx.reshape(-1,1)*n.reshape(1,3)
    return x+dx.reshape(-1,1)*n.reshape(1,3)
def add_photo_z_simple_proj1(pos, sigma_D, the_los):
    Delta = np.random.normal(loc=0.0, scale=sigma_D, size=len(pos[:,0]))
    print(Delta, np.cos(the_los))
    pos_new = np.zeros(pos.shape)
    pos_new[:,0]=pos[:,0]+Delta*np.cos(the_los)
    pos_new[:,1]=pos[:,1]+Delta*np.sin(the_los)
    pos_new[:,2]=pos[:,2]
    return pos_new
#======================   project     ======================
def proj1(f,the):
    grid=f.shape[0]
    s=np.sin(the)
    c=np.cos(the)

    f_los=(f[:,:,:,0]*c+f[:,:,:,1]*s)

    x=np.arange(grid).reshape(-1,1,1)
    y=np.arange(grid).reshape(1,-1,1)
    z=np.arange(grid).reshape(1,1,-1)
    a=np.arange(grid)

    rx=(c*x-s*y+0*z)%grid
    ry=(s*x+c*y+0*z)%grid
    rz=(0*x+0*y+z)%grid
    xcoor=np.zeros((len(rx.flatten()),3))
    xcoor[:,0]=rx.flatten()
    xcoor[:,1]=ry.flatten()
    xcoor[:,2]=rz.flatten()
    xcoor[np.where((xcoor>grid-1)&(xcoor<grid-0.5))]=grid-1
    xcoor[np.where(xcoor>=grid-0.5)]=0

    fintp=interpn(np.array([a,a,a]),f_los,xcoor).reshape(grid,grid,grid)

    return np.sum(fintp,axis=0)
def proj1_halo_coor(halox,the):
    grid=halox.shape[0]
    s=np.sin(the)
    c=np.cos(the)
    
    halox_new = np.zeros(halox.shape)
    halox_new[:,0] = c*halox[:,0]+s*halox[:,1]
    halox_new[:,1] = -s*halox[:,0]+c*halox[:,1]
    halox_new[:,2] = halox[:,2]
    
    return halox_new
#======================  S/N prediction   ======================
def SN_prediction_tomography(l, cor, PS_rec, f_sky=1, CMB_survey="Planck"):
    if CMB_survey=="Planck":
        ss = 1
    elif CMB_survey=="ACT":
        ss=0
    else:
        print("CMB survey error")
    lll,cl_cmb=cmbcl(lmax=15000)
    lll,cl_noise=cmbnoise(Delta_T=cmbsur[ss][0],sigma=cmbsur[ss][1],lmax=15000)
    f_cmb = interpolate.interp1d(lll, cl_cmb)
    f_noise = interpolate.interp1d(lll, cl_noise)

    l_array = (l[1:]+l[:-1])/2
    dl = l[1:] - l[:-1]
    PS_rec = (PS_rec[1:]+PS_rec[:-1])/2
    cor = (cor[1:]+cor[:-1])/2
    NS = np.sqrt((f_cmb(l_array)+f_noise(l_array))*PS_rec/cor**2/(2*l_array*dl*f_sky))
    SN = np.sqrt(np.sum(1/NS**2))
    
    return l_array, NS, SN


##======================   stack     ======================
def AP_l(l,the_ap):
#W_ap(l|theta_ap)
    def w_top(x):
        return j1(x)/x
    x=l*the_ap*np.pi/180/60 # no unit
    wap=4*(w_top(x)-w_top(np.sqrt(2)*x))
    wap[np.where(x==0)]=0
    return wap
def convel_map2d_AP(field2d, the_ap, grid, lscale): #the_ap:arcmin
    field=field2d.reshape(grid,grid)
    fieldk=np.fft.rfft2(field)
    kmod=fa.karrayfor2d(grid)
    kmod[0,0]=1
    
    lmod = lscale*kmod
    W_ap = AP_l(lmod, the_ap=the_ap)
    fieldk = fieldk*W_ap
    field_ap = np.fft.irfft2(fieldk)
    return field_ap
def resample_jackknift(label, N_jk):
    N=len(label)
    n = N//N_jk
    #print(n, n*N)
    label = label[:N_jk*n].reshape(n, N_jk)
    
    label_array =np.zeros((n*(N_jk-1), N_jk), dtype=np.int32)
    for i in range(N_jk):
        if i==0:
            label_array[:,i]=label[:,1:].flatten();continue
        if i==N_jk-1:
            label_array[:,i]=label[:,:-1].flatten();continue
            
        label_array[:(i)*n,i] = label[:,:i].flatten()
        label_array[(i)*n:,i] = label[:,i+1:].flatten()
    return label_array
def error_esti_jackknife(samples):
#samples.shape=(r_bin,N_bin)
	r_bin=samples.shape[0]
	N_bin=samples.shape[1]
	s_mean=np.mean(samples,axis=1)
	corv=np.zeros((r_bin,r_bin))
	for ii in range(r_bin):
		for jj in range(r_bin):
			for kk in range(N_bin):
				corv[ii,jj]+=(samples[ii,kk]-s_mean[ii])*(samples[jj,kk]-s_mean[jj])
	corv*=(N_bin-1.)/N_bin
	#esti=(N_bin-r_bin-1.)/(N_bin-1)*np.linalg.inv(corv)
	#print("esti:\n",esti)
	return s_mean,corv
def pseudo_inverse(C,n):
	N=C.shape[0]
	w,v =np.linalg.eig(C)
	w=np.sort(w)

	if n==0:
		return np.linalg.pinv(C)
	else:
		return np.linalg.pinv(C,(w[n]+w[n-1])/2/w[-1])
def cal_chi_square(v1,v2,C_):
	v1=np.mat(v1)
	v2=np.mat(v2)
	return np.dot(v1,np.dot(C_,v2.T))[0,0]
def cal_vel_rec_proj1(halox, vel_rec):
    halox_new = (proj1_halo_coor(halox,the_los)*grid)%grid
    vx = interpn((np.arange(Grid),np.arange(Grid),np.arange(Grid)), vel_rec[:,:,:,0], halox*Grid, bounds_error=0, fill_value=0)
    vy = interpn((np.arange(Grid),np.arange(Grid),np.arange(Grid)), vel_rec[:,:,:,1], halox*Grid, bounds_error=0, fill_value=0)
    halo_v_rec = np.cos(the)*vx+np.sin(the)*vy
    
    return halo_v_rec
def cal_stack_kSZ_signal_proj1_one_sample(CMB_map, the_ap, vel_rec, halox, halov_los, the_los, grid, thelen):
    #CMB_map: rsd_coor
    #vel_rec, halox: original coor
    #halox: [0,1]
    #the_ap: arcmin
    #thelen: rad
    #the_los: rad, rotate x-y plane and project along x-axis
    CMB_ap = convel_map2d_AP(field2d=CMB_map, the_ap=the_ap, grid=grid, lscale=2*np.pi/thelen)
    halox_new = (proj1_halo_coor(halox,the_los)*grid)%grid
    
    halo_kSZ = interpn((np.arange(grid), np.arange(grid)), CMB_ap, (halox_new[:,1], halox_new[:, 2]), bounds_error=0, fill_value=0)

    vx = interpn((np.arange(Grid),np.arange(Grid),np.arange(Grid)), velh_rsd_rec[:,:,:,0], halox*Grid, bounds_error=0, fill_value=0)
    vy = interpn((np.arange(Grid),np.arange(Grid),np.arange(Grid)), velh_rsd_rec[:,:,:,1], halox*Grid, bounds_error=0, fill_value=0)
    halo_v_rec = np.cos(the)*vx+np.sin(the)*vy
    
    a, b = halo_kSZ, halo_v_rec, 
    r_kSZ_rec = np.mean(a*b)/np.std(a)/np.std(b)
    a, b = halo_v_rec, halov_los
    r_true_rec = np.mean(a*b)/np.std(a)/np.std(b)

    stack_signal = np.sum(halo_kSZ*halo_v_rec)/np.sum(halo_v_rec**2)*np.std(halo_v_rec)*np.pi*the_ap**2*10**6/r_true_rec
    np.random.shuffle(halo_v_rec)
    stack_signal_null = np.sum(halo_kSZ*halo_v_rec)/np.sum(halo_v_rec**2)*np.std(halo_v_rec)*np.pi*the_ap**2*10**6/r_true_rec

    return stack_signal, stack_signal_null
def cal_stack_kSZ_signal_proj1_jk(the_ap, CMB_map, vel_rec, halox, halov, the_los, N_jk, thelen, grid,):
    '''
    vel_rec: (grid, grid, grid, 3), simu coor
    halox: (N, 3), simu coor
    halov: (N, 3), simu coor, only for rv
    the_los:
    '''
    Nh = len(halov[:,0])
    stack_signal = np.zeros((len(the_ap), N_jk))

    #vec_rec_los
    vx = interpn((np.arange(grid),np.arange(grid),np.arange(grid)), vel_rec[:,:,:,0], halox*grid, bounds_error=0, fill_value=0)
    vy = interpn((np.arange(grid),np.arange(grid),np.arange(grid)), vel_rec[:,:,:,1], halox*grid, bounds_error=0, fill_value=0)
    halov_rec_los = np.cos(the_los)*vx+np.sin(the_los)*vy

    halov_los = np.cos(the_los)*halov[:,0]+np.sin(the_los)*halov[:,1]
    r_true_rec = np.mean(halov_rec_los*halov_los)/np.std(halov_rec_los)/np.std(halov_los)
    del halov_los, halov

    #kSZ temp
    halox_new_rsd_coor = (proj1_halo_coor(halox,the_los)*grid)%grid
    #jk
    label_array = resample_jackknift(label=np.arange(len(halov_rec_los)), N_jk=N_jk)
    r_ksz_rec = np.zeros(len(the_ap))
    print("rrv=",r_true_rec)
    for i in (range(len(the_ap))):
        CMB_ap_map = convel_map2d_AP(field2d=CMB_map, the_ap=the_ap[i], grid=grid, lscale=2*np.pi/thelen)
        kSZ_ap_halo = interpn((np.arange(grid), np.arange(grid)), CMB_ap_map, (halox_new_rsd_coor[:,1], halox_new_rsd_coor[:, 2]), bounds_error=0, fill_value=0)
        r_ksz_rec[i] = np.mean(halov_rec_los*kSZ_ap_halo)/np.std(halov_rec_los)/np.std(kSZ_ap_halo)

        stack_signal[i,:] = np.sum(kSZ_ap_halo[label_array]*halov_rec_los[label_array], axis=0)/np.sum(halov_rec_los[label_array]**2, axis=0)*np.std(halov_rec_los[label_array], axis=0)*np.pi*the_ap[i]**2*10**6/r_true_rec

    return stack_signal, r_true_rec, r_ksz_rec

def cal_stack_kSZ_signal_proj1_jk_mass_weight(the_ap, CMB_map, vel_rec, halox, halov, halom, the_los, N_jk, thelen, grid,):
    '''
    vel_rec: (grid, grid, grid, 3), simu coor
    halox: (N, 3), simu coor
    halov: (N, 3), simu coor, only for rv
    the_los:
    '''
    Nh = len(halov[:,0])
    stack_signal = np.zeros((len(the_ap), N_jk))

    #vec_rec_los
    vx = interpn((np.arange(grid),np.arange(grid),np.arange(grid)), vel_rec[:,:,:,0], halox*grid, bounds_error=0, fill_value=0)
    vy = interpn((np.arange(grid),np.arange(grid),np.arange(grid)), vel_rec[:,:,:,1], halox*grid, bounds_error=0, fill_value=0)
    halov_rec_los = np.cos(the_los)*vx+np.sin(the_los)*vy

    halov_los = np.cos(the_los)*halov[:,0]+np.sin(the_los)*halov[:,1]
    r_true_rec = np.mean(halov_rec_los*halov_los)/np.std(halov_rec_los)/np.std(halov_los)
    del halov_los, halov
    print("rrv=",r_true_rec)

    #kSZ temp
    halox_new_rsd_coor = (proj1_halo_coor(halox,the_los)*grid)%grid
    print("rrv=",r_true_rec)
    #jk
    label_array = resample_jackknift(label=np.arange(len(halov_rec_los)), N_jk=N_jk)
    print("rrv=",r_true_rec)
    r_ksz_rec = np.zeros(len(the_ap))
    print("rrv=",r_true_rec)
    for i in (range(len(the_ap))):
        print(i)
        CMB_ap_map = convel_map2d_AP(field2d=CMB_map, the_ap=the_ap[i], grid=grid, lscale=2*np.pi/thelen)
        kSZ_ap_halo = interpn((np.arange(grid), np.arange(grid)), CMB_ap_map, (halox_new_rsd_coor[:,1], halox_new_rsd_coor[:, 2]), bounds_error=0, fill_value=0)
        r_ksz_rec[i] = np.mean(halov_rec_los*kSZ_ap_halo)/np.std(halov_rec_los)/np.std(kSZ_ap_halo)

        stack_signal[i,:] = np.sum(halom[label_array]*np.abs(halov_rec_los[label_array])*kSZ_ap_halo[label_array]*halov_rec_los[label_array], axis=0)/np.sum(halom[label_array]*halov_rec_los[label_array]**2, axis=0)*np.pi*the_ap[i]**2*10**6/r_true_rec

    return stack_signal, r_true_rec, r_ksz_rec
