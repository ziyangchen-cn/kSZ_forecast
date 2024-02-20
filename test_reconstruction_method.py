from kSZ_forecast_general_func import *

# parameters
redshift_bin = int(sys.argv[1])
Grid = int(sys.argv[2])
MAS = sys.argv[3]
galaxy_number_density_label = 2
HOD_random_seed = 100


the_los=(45)*np.pi/180
n_rsd=np.array([np.cos(the_los),np.sin(the_los),0])

def decompose_along_n_vel(vel, n_los):
    grid = vel.shape[0]
    vel_rec_los = np.sum(vel.reshape(grid, grid, grid, 3)*n_los.reshape(1, 1, 1, 3), axis = 3)
    return vel_rec_los
def cal_r_halo_sample_proj1(vel_rec, halox, halov, halo_mass, the_los, grid):
    halov_los = np.cos(the_los)*halov[:,0]+np.sin(the_los)*halov[:,1]
    r = np.zeros(len(lgM_min))
    sigma_v_rec = np.zeros(len(lgM_min))
    for i in range(len(lgM_min)):
        lgM_min2 = lgM_min[i]
        if i == 0:
            lgM_max2 = 20
        else:
            lgM_max2 = lgM_min[i-1]
        label = np.where((halo_mass<10**lgM_max2)&(halo_mass>10**lgM_min2))[0]
    
        vx = interpn((np.arange(grid),np.arange(grid),np.arange(grid)), vel_rec[:,:,:,0], halox[label]*grid, bounds_error=0, fill_value=0)
        vy = interpn((np.arange(grid),np.arange(grid),np.arange(grid)), vel_rec[:,:,:,1], halox[label]*grid, bounds_error=0, fill_value=0)
        halov_rec_los = np.cos(the_los)*vx+np.sin(the_los)*vy

        
        r[i] = np.mean(halov_rec_los*halov_los[label])/np.std(halov_rec_los)/np.std(halov_los[label])
        sigma_v_rec[i] = np.std(halov_rec_los)
    return r, sigma_v_rec

if redshift_bin == 0:
    Snapshot = 2448
    lgM_min = np.array([13.21616884, 13.00613391, 12.87405033, 12.6986018 , 12.56470822, 12.47015211, 11.99107016, 11.62894912])
    n_gal = np.array([4, 4.4, 4.8])*10**-4
    HOD_model = "DESI"
    gll = [0,1,2]
    rr = "0.6 < z < 1"
    z_eff = 0.8
if redshift_bin == 1:
    Snapshot = 2448
    lgM_min = np.array([13.21616884, 13.00613391, 12.87405033, 12.6986018 , 12.56470822, 12.47015211, 11.99107016, 11.62894912])
    n_gal = np.array([6, 6.6, 7.2])*10**-4
    HOD_model = "HSC_NB816"
    gll = [0,1,2]
    rr = "1 < z < 1.6"
    z_eff = 1.3
if redshift_bin == 2:
    Snapshot = 1631
    lgM_min = np.array([12.84230083, 12.66537838, 12.54430498, 12.39486591, 12.29210015, 12.2125257 , 11.80201392, 11.48552698])
    n_gal = np.array([3, 3.3, 3.6])*10**-4
    HOD_model = "HSC_NB912"
    gll = [0,1,2]
    rr = "1.6 < z < 2.4"
    z_eff = 2.0

print("Read halos")
halox,halov, mh =ReadHalos_np(zstep=Snapshot)
halo_mass = mh*pm
haloxx_rsd=add_RSD(halox*Boxlen,halov*(1+z_eff), n_rsd, cosmo.H(z_eff).value)%Boxlen   #Mpc/h

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
galaxy_pos_rsd = np.zeros((N_gal, 3))
for i in range(3):
    galaxy_pos_rsd[:,i]=np.repeat(haloxx_rsd[:,i], richness)
    galaxy_pos[:,i]=np.repeat(halox[:,i], richness)

#rand to make density (for n_gal)
N_gal_need = np.int32(n_gal[galaxy_number_density_label]*Boxlen**3)
a = np.arange(N_gal, dtype=np.int32)
np.random.shuffle(a)
galaxy_pos = galaxy_pos[list(a[:N_gal_need]),:]
galaxy_pos_rsd = galaxy_pos_rsd[list(a[:N_gal_need]),:]
N_gal = len(galaxy_pos[:,0])
print("Number of galaxies", N_gal,"   n=",N_gal/Boxlen**3)

print("Read Den/Mom")
cnorm=ne0*sigma_t*Mpc2m*Boxlen*1.0/Grid*(1+z_eff)**2/h  # proj mom: km/h -> dkSZ/CMB
Den_dir= "/home/chenzy/data/denmap/denNGP_"+str(Grid)+"_"+str(Snapshot)+"_0"
Mom_dir = "/home/chenzy/data/mommap/momNGP_"+str(Grid)+"_"+str(Snapshot)+"_0"
momp=snap_info.get_vel_norm(Snapshot)*np.fromfile(Mom_dir, dtype=np.float32, count=3*Grid**3, sep="").reshape(Grid,Grid,Grid,3)
denp=np.fromfile(Den_dir, dtype=np.float32, count=Grid**3, sep="").reshape((Grid,)*3)
velp = momden2vel(denp, momp)
momp = (denp-1).reshape(Grid, Grid, Grid,1)*velp
momp_proj=proj1(momp,the_los)*cnorm

if MAS == "NGP":
	deng_rsd = den_NGP(galaxy_pos_rsd,Grid)
	deng = den_NGP(galaxy_pos,Grid)
if MAS == "CIC":
	deng_rsd = den_CIC(x=galaxy_pos_rsd, grid = Grid,boxlen=Boxlen)
	deng = den_CIC(x=galaxy_pos, grid = Grid,boxlen=Boxlen)
if MAS == "TSC":
	deng_rsd = den_TSC(x=galaxy_pos_rsd, grid = Grid,boxlen=Boxlen)
	deng = den_TSC(x=galaxy_pos, grid = Grid,boxlen=Boxlen)

#from denp
theta = den2theta(den=denp, redshift=z_eff, cosmo_model=cosmo, grow_rate_index=4/7.)
vel_rec_0 = theta2vel(theta, Boxlen)
#no rsd
theta = den2theta(den=deng, redshift=z_eff, cosmo_model=cosmo, grow_rate_index=4/7.)
vel_rec_no_rsd = theta2vel(theta, Boxlen)
#no filter
theta = den2theta(den=deng_rsd, redshift=z_eff, cosmo_model=cosmo, grow_rate_index=4/7.)
vel_rec_1 = theta2vel(theta, Boxlen)
#f_g
vel_rec_g = []

for Rg  in [1, 2, 3,4, 5,6, 7, 10]:
    den_fg = den_filter_g(den=deng_rsd, Rg=Rg, Boxlen=Boxlen)
    theta = den2theta(den=den_fg, redshift=z_eff, cosmo_model=cosmo, grow_rate_index=4/7.)
    vel_rec_2 = theta2vel(theta, Boxlen)
    vel_rec_g.append(vel_rec_2)
#f_w
thetap = vel2theta(velp, Boxlen)
W,kx,ky,kz=wiener_filter_theta_esti(deng_rsd,thetap, Boxlen)
thetag_rsd_rec=wiener_filter_den2theta(deng_rsd,W,kx,ky,kz, Boxlen)
vel_rec_w=theta2vel(thetag_rsd_rec, Boxlen)
r_halo_rec_w, sigma_v_rec_w = cal_r_halo_sample_proj1(vel_rec=vel_rec_w, halox=halox, halov=halov, halo_mass=halo_mass, the_los=the_los, grid=Grid)

r_halo_rec_no_rsd, sigma_v_rec_no_rsd = cal_r_halo_sample_proj1(vel_rec=vel_rec_no_rsd, halox=halox, halov=halov, halo_mass=halo_mass, the_los=the_los, grid=Grid)
r_halo_rec_1, sigma_v_rec_1 = cal_r_halo_sample_proj1(vel_rec=vel_rec_1, halox=halox, halov=halov, halo_mass=halo_mass, the_los=the_los, grid=Grid)
r_halo_rec_0, sigma_v_rec_0 = cal_r_halo_sample_proj1(vel_rec=vel_rec_0, halox=halox, halov=halov, halo_mass=halo_mass, the_los=the_los, grid=Grid)

r_halo_rec_g = np.zeros((len(lgM_min), len(vel_rec_g)))
sigma_v_rec_g = np.zeros((len(lgM_min), len(vel_rec_g)))
for i in range(len(vel_rec_g)):
    r_halo_rec_g[:,i], sigma_v_rec_g[:,i] = cal_r_halo_sample_proj1(vel_rec=vel_rec_g[i], halox=halox, halov=halov, halo_mass=halo_mass, the_los=the_los, grid=Grid)

#r(k)
velp_los = decompose_along_n_vel(vel=velp, n_los=n_rsd)
vel_rec_0_los  = decompose_along_n_vel(vel=vel_rec_0, n_los=n_rsd)
vel_rec_1_los  = decompose_along_n_vel(vel=vel_rec_1, n_los=n_rsd)
vel_rec_w_los  = decompose_along_n_vel(vel=vel_rec_w, n_los=n_rsd)
vel_rec_no_rsd_los  = decompose_along_n_vel(vel=vel_rec_no_rsd, n_los=n_rsd)
rk_no_rsd, k = fa.CalR(vel_rec_no_rsd_los, velp_los, Grid)
rk_0, k = fa.CalR(vel_rec_0_los, velp_los, Grid)
rk_1, k = fa.CalR(vel_rec_1_los, velp_los, Grid)
rk_w, k = fa.CalR(vel_rec_w_los, velp_los, Grid)

rk_g = np.zeros((len(k), len(vel_rec_g)))
for i in range(len(vel_rec_g)):
	vel_rec_los  = decompose_along_n_vel(vel=vel_rec_g[i], n_los=n_rsd)
	rk_g[:, i], k = fa.CalR(vel_rec_los, velp_los, Grid)

filename="test_rec_method/test_redshiftbin"+str(redshift_bin)+"_"+str(Grid)+"_"+MAS+".npy"
np.save(filename, (r_halo_rec_0, sigma_v_rec_0, r_halo_rec_1, sigma_v_rec_1, r_halo_rec_no_rsd, sigma_v_rec_no_rsd, r_halo_rec_g,sigma_v_rec_g, r_halo_rec_w, sigma_v_rec_w, k, rk_g, rk_no_rsd, rk_0, rk_1, rk_w))









