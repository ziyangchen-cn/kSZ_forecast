import numpy as np
import fftanalysis as fa
import time
from scipy import interpolate
k_cut=30

zvalue={99:0, 91:0.1, 84:0.2, 78:0.3, 72:0.4, 67:0.5, 59:0.7, 50:1, 40:1.5, 33:2, 25:3.01, 21:4.01, 17:5, 13:6, 11:7, 8:8, 6:9, 4:10, 3:10.98, 2:11.98}
zstep=[99,91,84,78, 72, 67, 59, 50 ,40, 33, 25, 21, 17, 13, 11, 8, 6, 4, 3, 2]


def get_ampthemom(zp):
# for a giving z, output ampmom themom
	workdir="/home/chenzy/data/reconstpara/rpmom"

	# ensure 2 zstep files to interpole
	for i in range(len(zstep)):
		if zvalue[zstep[i]]>zp: break;

	#weights 1 i-1 	2 i
	w1=(zp-zvalue[zstep[i]])/(zvalue[zstep[i-1]]-zvalue[zstep[i]])
	w2=(zvalue[zstep[i-1]]-zp)/(zvalue[zstep[i-1]]-zvalue[zstep[i]])

	#read 2 files
	k=np.load(workdir+str(zstep[i-1])+".npz")["k"]
	ampmom1=np.load(workdir+str(zstep[i-1])+".npz")["amp"]
	themom1=np.load(workdir+str(zstep[i-1])+".npz")["the"]

	ampmom2=np.load(workdir+str(zstep[i])+".npz")["amp"]
	themom2=np.load(workdir+str(zstep[i])+".npz")["the"]
	
	ampmom=w1*ampmom1+w2*ampmom2
	themom=w1*themom1+w2*themom2

	print ("\t The range of k: ", k[0],'~',k[-1],'h/Mpc')

	return k, ampmom, themom

def ChangeAmp(mapk,kmod,amp,kamp,kscale):
#kmod is in grid unit
#kamp h/Mpc
	precise=100
	amp2=np.zeros(np.int((np.max(kmod)+1)*precise))
	def interpamp(k):
		if (k<kamp[0]):return 1
		if (k>kamp[-1]):return amp[-1]
		for i in range(len(kamp)):
			if k<kamp[i]:break
		return amp[i-1]+(k-kamp[i-1])/(kamp[i]-kamp[i-1])*(amp[i]-amp[i-1])
	for i in range(len(amp2)):
		k=(1+i*1.0/precise)*kscale
		amp2[i]=interpamp(k)
	amp3=amp2[np.round((kmod[1:,1:,1:]-1)*precise).astype(np.int)]
	mapk[1:,1:,1:]=amp3*mapk[1:,1:,1:]

def ChangeTheta(mapk,kmod,theta,kthe,kscale,k_cut):
# theta is sigma
	precise=1000
	theta2=np.zeros(np.int((np.max(kmod)+1)*precise))
	# theta2 is determine the precise of interp
	# label 0 represent k=1grid: k=1+label/precise
	def interptheta(k):
		# here k is of the physical unit
		if (k<kthe[0]):return 0
		if (k>=k_cut):return 0
		#go beyond the range of given data
		for i in range(len(kthe)):
			if k<kthe[i]:break
		return theta[i-1]+(k-kthe[i-1])/(kthe[i]-kthe[i-1])*(theta[i]-theta[i-1])
	for i in range(len(theta2)):
	# interp theta2
		k=(1+i*1.0/precise)*kscale
		theta2[i]=interptheta(k)
		#print k, theta2[i]
	theta3=theta2[np.round((kmod[1:,1:,1:]-1)*precise).astype(np.int)]*np.random.randn(kmod.shape[0]-1,kmod.shape[1]-1,kmod.shape[2]-1)
	#theta3 is the same shape of mapk[1:,1:,1:]
	mapk[1:,1:,1:]=mapk[1:,1:,1:]*np.exp(1*theta3*1j)


def ReconField(field,grid, boxlen,amp=False, theta=False, kamp=False,kthe=False):
# unit of kamp & kthe is h/Mpc
	field=field.reshape(grid,grid,grid)
	mapk=np.fft.rfftn(field)
	t0=time.time()
	print("\t==================================")
	print("\tReconstruct a field")
	kmod=fa.karray(grid)
	if type(amp)==bool:
		print("\tAmplitude is False")
	else:
		print("\tAmplitude is True")
		ChangeAmp(mapk,kmod,amp,kamp, 2*np.pi/boxlen)

	if type(theta)==bool:
		print("\tTheta is False")
	else:
		print("\tTheta is True")
		#Only modify when r>r_cut
		k_c=k_cut/10.0
		for kno in range(len(kthe)):
			if kthe[kno]>k_c:break
		the_cut=theta[kno-1]+(k_c-kthe[kno-1])/(kthe[kno]-kthe[kno-1])*(theta[kno]-theta[kno-1])
		r_cut=np.exp(-the_cut**2/2)
		#r_cut=0.9
		#the_cut=np.sqrt(2-2*r_cut)
		'''
		for kno in range(len(kthe)):
			if theta[kno]>the_cut:break
		k_cut=kthe[kno-1]+(the_cut-theta[kno-1])/(theta[kno]-theta[kno-1])*(kthe[kno]-kthe[kno]
		'''
		print ('\t','r_cut=',r_cut,'\ttheta_cut=',the_cut,'\tk_cut=',k_c)
		ChangeTheta(mapk, kmod, theta, kthe, 2*np.pi/boxlen,k_c)
	print("\tReconstruct a field cost   "+str(np.round(time.time()-t0,3))+'s')

	field=np.fft.irfftn(mapk)

	return field

def ReconField1(field,grid, boxlen,amp=False, theta=False, kamp=False,kthe=False):
# unit of kamp & kthe is h/Mpc
# different from up: interpolation
	field=field.reshape(grid,grid,grid)
	mapk=np.fft.rfftn(field)
	t0=time.time()
	print("\t==================================")
	print("\tReconstruct a field")
	kmod=fa.karray(grid)
	if type(amp)==bool:
		print("\tAmplitude is False")
	else:
		print("\tAmplitude is True")
		inp_amp=interpolate.interp1d(kamp,amp,bounds_error=False, fill_value=1)
		mapk[:,:,:]=mapk[:,:,:]*inp_amp(kmod[:,:,:]*2*np.pi/boxlen)
		del inp_amp

	if type(theta)==bool:
		print("\tTheta is False")
	else:
		print("\tTheta is True")
		#Only modify when r>r_cut
		k_c=k_cut/10.0
		for kno in range(len(kthe)):
			if kthe[kno]>k_c:break
		the_cut=theta[kno-1]+(k_c-kthe[kno-1])/(kthe[kno]-kthe[kno-1])*(theta[kno]-theta[kno-1])
		r_cut=np.exp(-the_cut**2/2)
		#r_cut=0.9
		#the_cut=np.sqrt(2-2*r_cut)
		'''
		for kno in range(len(kthe)):
			if theta[kno]>the_cut:break
		k_cut=kthe[kno-1]+(the_cut-theta[kno-1])/(theta[kno]-theta[kno-1])*(kthe[kno]-kthe[kno]
		'''
		intp_the=interpolate.interp1d(kthe,theta,bounds_error=False, fill_value=0)
		the_k=intp_the(kmod*2*np.pi/boxlen)*np.random.randn(kmod.shape[0],kmod.shape[1],kmod.shape[2])
		mapk[:,:,:]=mapk[:,:,:]*np.exp(1*the_k*1j)
		del the_k
	print("\tReconstruct a field cost   "+str(np.round(time.time()-t0,3))+'s')

	field=np.fft.irfftn(mapk)

	return field

def ModifyMom(mom,z,grid,boxlen):
	t0=time.time()
	k,ampmom,themom=get_ampthemom(z)
	mompb=np.zeros((grid,grid,grid,3))
	mompb[:,:,:,0]=ReconField1(mom[:,:,:,0],grid,boxlen,amp=ampmom,theta=themom,kamp=k,kthe=k)
	mompb[:,:,:,1]=ReconField1(mom[:,:,:,1],grid,boxlen,amp=ampmom,theta=themom,kamp=k,kthe=k)
	mompb[:,:,:,2]=ReconField1(mom[:,:,:,2],grid,boxlen,amp=ampmom,theta=themom,kamp=k,kthe=k)
	print("Motify mom field of baryon......",np.round(time.time()-t0,3))
	return mompb

