from kSZ_forecast_general_func import *

Grid = int(sys.argv[1])
redshift_bin = int(sys.argv[2])
den_method = int(sys.argv[3])   #NGP
vel_method = int(sys.argv[4])   #thetap
HOD_input = sys.argv[5]
HOD_random_seed = int(sys.argv[6])

CMB_survey = "ACT"
MAS = "NGP"
the_los=(45)*np.pi/180
n_rsd=np.array([np.cos(the_los),np.sin(the_los),0])
np.random.seed(HOD_random_seed)

if redshift_bin == 0:
	Snapshot = 2746#2448
	n_gal = np.array([4, 4.4, 4.8])*10**-4
	HOD_model = "DESI"
	gll = [0,1,2]
	rr = "0.6 < z < 1"
	z_eff = 0.8
	bias_g = 1.173
if redshift_bin == 1:
	Snapshot = 2181#2448
	n_gal = np.array([6, 6.6, 7.2])*10**-4
	HOD_model = "HSC_NB816"
	gll = [3, 4, 5]
	rr = "1 < z < 1.6"
	z_eff = 1.3
	bias_g = 1.703
if redshift_bin == 2:
	Snapshot = 1631
	n_gal = np.array([3, 3.3, 3.6])*10**-4
	HOD_model = "HSC_NB912"
	gll = [0,1,2]
	rr = "1.6 < z < 2.4"
	z_eff = 2.0
	bias_g = 2.739
if not HOD_input=="0":
   HOD_model = HOD_input


Thelen = Boxlen/h/cosmo.comoving_distance(z=z_eff).value
cnorm=ne0*sigma_t*Mpc2m*Boxlen*1.0/Grid*(1+z_eff)**2/h  # proj mom: km/h -> dkSZ/CMB

print("read DEN, MOM")
Den_dir= "/home/chenzy/data/denmap/den"+MAS+"_"+str(Grid)+"_"+str(Snapshot)+"_0"
Mom_dir = "/home/chenzy/data/mommap/mom"+MAS+"_"+str(Grid)+"_"+str(Snapshot)+"_0"
momp=snap_info.get_vel_norm(Snapshot)*np.fromfile(Mom_dir, dtype=np.float32, count=3*Grid**3, sep="").reshape(Grid,Grid,Grid,3)
denp=np.fromfile(Den_dir, dtype=np.float32, count=Grid**3, sep="").reshape((Grid,)*3)
velp = momden2vel(denp, momp)
momp = (denp-1).reshape(Grid, Grid, Grid,1)*velp
momp_proj=proj1(momp,the_los)*cnorm

print("read halo")
halox,halov, mh =ReadHalos_np(zstep=Snapshot)
halo_mass = mh*pm

haloxx_rsd=add_RSD(halox*Boxlen,halov*(1+z_eff), n_rsd, cosmo.H(z_eff).value)%Boxlen   #Mpc/h
#haloxx_rsd=add_RSD(halox*Boxlen,halov*(1+snap_info.get_redshift(Snapshot)), n_rsd, cosmo.H(snap_info.get_redshift(Snapshot)).value)%Boxlen   #Mpc/h

pkden,kden=fa.CalPS(denp,Grid,mapscale=(Boxlen/Grid**2)**3,kscale=2*np.pi/Boxlen)
velp = momden2vel(denp, momp)
thetap = vel2theta(velp, Boxlen)

for i in [0,1,2]:
	# =========== galaxy mock sample ==============
	galaxy_number_density_label = i
	#halo sample for velocity reconstruction(for HOD_method)
	if HOD_model == "DESI":
		richness = np.zeros(len(halo_mass), dtype = np.int32)
		for i in range(4):
			HOD_model_i = "DESI_L"+str(i)
			d = np.load("./halo_Temporary_storage/halos_"+str(Snapshot)+"_"+HOD_model_i+"_seed"+str(HOD_random_seed)+".npz")
			N_cen = d["N_cen"]
			N_sat = d["N_sat"]
			richness+=(N_cen + N_sat)
	else:
		d = np.load("./halo_Temporary_storage/halos_"+str(Snapshot)+"_"+HOD_model+"_seed"+str(HOD_random_seed)+".npz")
		N_cen = d["N_cen"]
		N_sat = d["N_sat"]
		richness = N_cen + N_sat
	N_gal = np.sum(richness)
	print("Number of galaxies", N_gal,"   n=",N_gal/Boxlen**3)
	galaxy_pos = np.zeros((N_gal, 3))
	for i in range(3):
		galaxy_pos[:,i]=np.repeat(haloxx_rsd[:,i], richness)

	#rand to make density (for n_gal)
	N_gal_need = np.int32(n_gal[galaxy_number_density_label]*Boxlen**3)
	a = np.arange(N_gal, dtype=np.int32)
	np.random.shuffle(a)
	galaxy_pos = galaxy_pos[list(a[:N_gal_need]),:]
	N_gal = len(galaxy_pos[:,0])
	print("Number of galaxies", N_gal,"   n=",N_gal/Boxlen**3)
	#===============================================


	#========== reconstruct kSZ tamplate ===========
	deng_rsd=den_NGP(galaxy_pos,Grid)
	W,kx,ky,kz=wiener_filter_theta_esti(deng_rsd,thetap, Boxlen)
	#del thetap
	thetag_rsd_rec=wiener_filter_den2theta(deng_rsd,W,kx,ky,kz, Boxlen)
	#del W,kx,ky,kz
	velg_rsd_rec=theta2vel(thetag_rsd_rec, Boxlen)


	deng_rsd_fw = wiener_filter_den(f=deng_rsd, bg=bias_g, pk=pkden, k=kden, kscale=2*np.pi/Boxlen, n=n_gal[galaxy_number_density_label], grid=Grid)

	momg_rsd_rec=velg_rsd_rec*(deng_rsd_fw).reshape(Grid,Grid,Grid,1)
	momg_rsd_rec_proj = proj1(momg_rsd_rec, the_los)*cnorm
	#===============================================

	r_den, k = fa.CalR(denp, deng_rsd_fw, Grid)
	r_vel, k = fa.CalRvector(velp, velg_rsd_rec, Grid)
	r_mom, k = fa.CalRvector(momp, momg_rsd_rec, Grid)



	cor, l = fa.CalCor2d(momg_rsd_rec_proj, momp_proj, Grid, mapscale=(Thelen/Grid**2)**2,kscale=2*np.pi/Thelen)
	ps_rec, l = fa.CalPS2d(momg_rsd_rec_proj, Grid, mapscale=(Thelen/Grid**2)**2,kscale=2*np.pi/Thelen)
	ps, l = fa.CalPS2d(momp_proj, Grid, mapscale=(Thelen/Grid**2)**2,kscale=2*np.pi/Thelen)

	CMB_survey = "ACT"
	l_array, NS, SN = SN_prediction_tomography(l=l, cor=cor, PS_rec=ps_rec, f_sky=1400/(4*np.pi*(180/np.pi)**2), CMB_survey=CMB_survey)
	print("S/N", SN)

	filename = "tomography_prediction_results/RedshiftBin"+str(redshift_bin)+"S"+str(Snapshot)+"G"+str(Grid)+"_gal_"+HOD_model+"_den"+str(den_method)+"vel"+str(vel_method)+"l"+str(galaxy_number_density_label)+"_seed"+str(HOD_random_seed)
	np.savez(filename, l=l, cor=cor, ps_rec=ps_rec, ps=ps, l_array=l_array, NS=NS, SN=SN, r_den = r_den, r_vel=r_vel, r_mom=r_mom, k = k*2*np.pi/Boxlen)
