from  scipy.special import j1
from kSZ_forecast_general_func import *
n_halo =np.array([    1,     2,     3,     5,    7,     9,    30,   70])*10**-4 #8
lgM_min=np.array([12.86, 12.68, 12.55, 12.41, 12.3, 12.23, 11.82, 11.5])
snap_info=Snap_Info()

Grid = int(sys.argv[1])
redshift_bin = int(sys.argv[2])
galaxy_number_density_label = int(sys.argv[3])

mass_selection_label2 = int(sys.argv[4])
den_method = int(sys.argv[5])   #NGP
vel_method = int(sys.argv[6])   #thetap
HOD_input = sys.argv[7]
HOD_random_seed = int(sys.argv[8])

MAS = "NGP"
N_jk = 100
the_ap = np.linspace(1, 6, 10)

np.random.seed(HOD_random_seed)

if redshift_bin == 0:
    Snapshot = 2448
    n_gal = np.array([4, 4.4, 4.8])*10**-4
    HOD_model = "DESI" 
    gll = [3, 4, 5]
    rr = "0.6 < z < 1"
    cnorm=ne0*sigma_t*Mpc2m*Boxlen*1.0/Grid*(1+0.8)**2/h  # proj mom: km/h -> dkSZ/CMB
if redshift_bin == 1:
    Snapshot = 2448
    n_gal = np.array([6, 6.6, 7.2])*10**-4
    HOD_model = "HSC_NB816"
    gll = [0,1,2]
    rr = "1 < z < 1.6"
    cnorm=ne0*sigma_t*Mpc2m*Boxlen*1.0/Grid*(1+1.3)**2/h  # proj mom: km/h -> dkSZ/CMB
if redshift_bin == 2:
    Snapshot = 1631
    n_gal = np.array([3, 3.3, 3.6])*10**-4
    HOD_model = "HSC_NB912"
    gll = [0,1,2]
    rr = "1.6 < z < 2.4"
    cnorm=ne0*sigma_t*Mpc2m*Boxlen*1.0/Grid*(1+2)**2/h  # proj mom: km/h -> dkSZ/CMB

if not HOD_input=="0":
   HOD_model = HOD_input

print(HOD_input, HOD_model, cnorm)

Thelen = Boxlen/h/cosmo.comoving_distance(z=snap_info.get_redshift(Snapshot)).value
#cnorm=ne0*sigma_t*Mpc2m*Boxlen*1.0/Grid*(1+snap_info.get_redshift(Snapshot))**2/h  # proj mom: km/h -> dkSZ/CMB

the_los=(45)*np.pi/180
n_rsd=np.array([np.cos(the_los),np.sin(the_los),0])

print("Read Den/Mom")
Den_dir= "/home/chenzy/data/denmap/den"+MAS+"_"+str(Grid)+"_"+str(Snapshot)+"_0"
Mom_dir = "/home/chenzy/data/mommap/mom"+MAS+"_"+str(Grid)+"_"+str(Snapshot)+"_0"
momp=snap_info.get_vel_norm(Snapshot)*np.fromfile(Mom_dir, dtype=np.float32, count=3*Grid**3, sep="").reshape(Grid,Grid,Grid,3)
denp=np.fromfile(Den_dir, dtype=np.float32, count=Grid**3, sep="").reshape((Grid,)*3)
velp = momden2vel(denp, momp)
momp = (denp-1).reshape(Grid, Grid, Grid,1)*velp
momp_proj=proj1(momp,the_los)*cnorm

print("Read halos")
halox,halov, mh =ReadHalos_np(zstep=Snapshot)
halo_mass = mh*pm

haloxx_rsd=add_RSD(halox*Boxlen,halov*(1+snap_info.get_redshift(Snapshot)), n_rsd, cosmo.H(snap_info.get_redshift(Snapshot)).value)%Boxlen   #Mpc/h
    
#halo sample for velocity reconstruction(for HOD_method)
print("Generate galaxy")
if HOD_model == "DESI":
    richness = np.zeros(len(halo_mass), dtype = np.int32)
    for i in range(4):
        HOD_modeli = "DESI_L"+str(i)
        richness+=np.load("./halo_Temporary_storage/halos_"+str(Snapshot)+"_richness_"+HOD_modeli+".npy")
else:
    richness = np.load("./halo_Temporary_storage/halos_"+str(Snapshot)+"_richness_"+HOD_model+".npy")
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
# for den_method
if den_method == 0:
    deng_rsd=den_NGP(galaxy_pos,Grid)
    
#for vel_method
if vel_method == 0:
    thetap = vel2theta(velp, Boxlen)
    W,kx,ky,kz=wiener_filter_theta_esti(deng_rsd,thetap, Boxlen)
    #del thetap
    thetag_rsd_rec=wiener_filter_den2theta(deng_rsd,W,kx,ky,kz, Boxlen)
    #del W,kx,ky,kz
    velg_rsd_rec=theta2vel(thetag_rsd_rec, Boxlen)


lgM_min2 = lgM_min[mass_selection_label2]
label2 = np.where(halo_mass>10**lgM_min2)[0]
mass_weight = halo_mass[label2]/np.mean(halo_mass[label2])
n_halo2 = n_halo[mass_selection_label2]
print("Halo sample to stack: n = "+str(n_halo2*10**4)+"$10^{-4}$ lgM_min "+str(lgM_min2))
#'''
for k in range(50):
	filename = "/home/chenzy/code/kSZ_forecast/stack_prediction_results/HOD_random_S"+str(Snapshot)+"G"+str(Grid)+"_gal_"+HOD_model+"_den"+str(den_method)+"vel"+str(vel_method)+"l"+str(gll[galaxy_number_density_label])+"l"+str(mass_selection_label2)+"ACT"+str(k).zfill(3)+"_seed"+str(HOD_random_seed)
	print(filename)
	cmb_map_act = np.load("/home/chenzy/code/kSZ_forecast/CMB_maps/ACT_"+str(Snapshot)+"_"+str(k).zfill(3)+".npy")
	stack_signal, r_true_rec, r_ksz_rec = cal_stack_kSZ_signal_proj1_jk(the_ap = the_ap, CMB_map = momp_proj+cmb_map_act, vel_rec=velg_rsd_rec, halox=halox[label2,:], halov=halov[label2, :], the_los=the_los, N_jk=N_jk, thelen=Thelen, grid = Grid)
	stack_mean, corv = error_esti_jackknife(stack_signal)
	C_ = pseudo_inverse(corv,2)
	chi_null = cal_chi_square(stack_mean, stack_mean, C_)

	print("S/N=",np.round(np.sqrt(chi_null), 2))
	np.savez(filename, stack_signal=stack_signal, r_true_rec =r_true_rec, r_ksz_rec=r_ksz_rec)
	#break

