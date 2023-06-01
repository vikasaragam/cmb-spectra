import numpy as np
# from numpy import array as arr
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


#%% Define model parameters

Mp    = 1 # reduced Planck mass

eH    = 10**-2 # slow-roll parameter \epsilon_H = - H'/H; assume constant
P0cmb = 2.1*10**(-9) # at CMB pivot scale k = 0.05 Mpc^{-1}
Htest = np.sqrt(8 * np.pi**2 * eH * Mp**2 * P0cmb)
Hcmb  = Htest / np.sqrt(2)

w0    = 0 # dimensionless turning rate max value; set nonzero to couple adiabatic and entropic modes
xi    = -3 # entropic mass matrix element M_{ss} = \xi \omega^2


#%% Define key quantities for CMB power spectra

MpcInv_to_Mp     = 2.6245e-57 # Mpc^{-1} / Mp 
kcmb             = 0.05 # CMB scalar pivot scale in Mpc^{-1}; 0.002 for tensor spectrum pivot scale
Ncmb_exit_to_end = 55 # how long before end of inflation does kcmb exit the horizon; a.k.a. Ncmb_e2e for future reference


Ncmb_i_to_exit   = 8 # number of e-folds before horizon exit at which CMB mode is initialized; must be large enough for Bunch-Davies initial conditions to hold, usually ~8-10 e-folds
Ncmb_i_to_end    = Ncmb_i_to_exit + Ncmb_exit_to_end # number of e-folds between initialization and end of inflation


deltaNcmb        = 2 # Ncmb_i_to_exit - deltaNcmb should not go lower than ~8 to ensure first mode to exit horizon is initialized sufficiently early
num_k_cmb        = 8 * deltaNcmb + 1 # odd number since Ncmb is at center of Ncmb_list
# Ncmb_list        = np.linspace(Ncmb - deltaNcmb, Ncmb + deltaNcmb, num_k_cmb)# list(range(Ncmb - deltaNcmb, Ncmb + deltaNcmb + 1))
deltaNk_cmb_list = np.linspace(-deltaNcmb, deltaNcmb, num_k_cmb) # deltaN = how long do other modes exit horizon after kcmb; deltaN < 0 exits before kcmb
Nk_cmb_e2e_list  = -deltaNk_cmb_list + Ncmb_exit_to_end

Hcmb_i           = Hcmb * np.exp(eH * Ncmb_i_to_exit) # from H' = - e_H * H => H = H_i e^{-e_H( N - N_i )}


#%% Define key time coordinates in e-folds

# Note that throughout this code, "N" refers to e-folds, i.e. dN = H dt such that a(N) = a_i e^N

# Quantities related to the perturbations' evolution:

# Ntot           = 65 # total number of e-folds of inflation from background evolution
# Npert_i        = 0 # e-fold coordinate at which perturbations' evolution begins
# Npert_f        = Ncmb_i_to_exit + 5 # e-fold coordinate at which perturbations' evolution ends
# Npert_tot      = Npert_f - Npert_i # total number of e-folds of perturbations' evolution

# Nk_exit_to_end = 55 # number of e-folds before end of inflation at which scale k exits horizon
# Nk_exit        = Ntot - Nk_exit_to_end


# Quantities related to the turning rate feature/profile switching on/off:

Nfeat_to_end   = 55 # number of e-folds of center of turning rate profile before end of inflation
Nfeat          = Ncmb_i_to_end - Nfeat_to_end # e-fold coordinate of center of feature, measured from initialization of CMB mode
deltaNfeat     = 0.25 # half-width of feature
Nfeat_i        = Nfeat - deltaNfeat
Nfeat_f        = Nfeat + deltaNfeat


#%% Define key functions and equations of motion

def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

def aN(a_i, N):
    # scale factor a(N) as function of e-folds N after a = a_i
    return a_i * np.exp(N)

def a_ini_cmb(kcmb_MpcInv, Ncmb_ini_to_hExit, Hub_exit):
    # Compute initial value of scale factor for CMB pivot scale
    log_a_init = np.log(kcmb_MpcInv) + np.log(MpcInv_to_Mp) - np.log(Hub_exit) - Ncmb_ini_to_hExit
    a_init     = np.exp(log_a_init)
    return a_init

def H_eom(IC, N, eps):
    # Equation of motion for Hubble parameter: H' = - eps * H, where prime denotes d/dN
    H = IC
    Hp = - eps * H
    return Hp

def logk_from_aH(a_i, Nk_ini_to_hExit, Hk_exit):
    # Use horizon exit condition, k = (aH)_exit
    logk = np.log(a_i) + Nk_ini_to_hExit + np.log(Hk_exit)
    return logk

def kaH2(N, Nk_ini_to_hExit, Hk_exit, Hi):
    # (k/(aH))^2; assume perfect de Sitter for a(N)
    # Nk_ini_to_hExit = number of e-folds from mode initialization to horizon exit, NOT number of e-folds before end of inflation!
    # Note that we avoid calling k_from_aH because the scale factor's initial (and extremely small) value divides out, which avoids potential numerical issues
    return np.exp( 2 * (Nk_ini_to_hExit - N) ) * (Hk_exit / Hi)**2

# def a_ini_k(kcmb_MpcInv, Nk_ini_to_hExit, Hub_exit, deltaN_k_to_cmb):
#     # Compute initial value of scale factor for mode initialized/exiting delta N e-folds after CMB pivot scale
#     # deltaN_k_to_cmb < 0 exits before kcmb
#     return a_ini_cmb(kcmb_MpcInv, Nk_ini_to_hExit, Hub_exit) * np.exp(deltaN_k_to_cmb)

# def k_from_deltaN(kcmb_MpcInv, Nk_ini_to_hExit, Hub_exit, deltaN_k_to_cmb):
#     a_i_k = a_ini_k(kcmb_MpcInv, Nk_ini_to_hExit, Hub_exit, deltaN_k_to_cmb)
#     return aN(a_i_k, Nk_ini_to_hExit) * Hub_exit

def w(wmax, N, Nfeat_center, deltaNfeat):
    # turning rate profile as function of e-fold time; uses difference of tanh functions to approximate top-hat profile
    Ni = Nfeat_center - deltaNfeat
    Nf = Nfeat_center + deltaNfeat
    m  = 2 # sets steepness of feature; higher value more closely approximates top-hat but leads to numerical precision issues in odeint
    # return wmax * ( np.heaviside(N-Ni, wmax/2) - np.heaviside(N-Nf, wmax/2) )
    return wmax * ( np.tanh( m * (N - Ni) ) - np.tanh( m * (N - Nf) ) ) / 2

def ms2H2(xi_ss, wmax, N, Nfeat_center, deltaNfeat):
    # 2012.02761 uses (xi-1) in their parametrization, since ms^2 = M_{ss} - w^2
    return (xi_ss-1) * w(wmax, N, Nfeat_center, deltaNfeat)**2

def eoms(IC, N, Nk_ini_to_hExit, Hk_exit, Hi, eps, xi_ss, wmax, Nfeat_center, deltaNfeat):
    # Here we follow the equations of motion given in 2012.02761
    # z, zp = IC
    z, zp, Qs, Qsp = IC
    zpp  = -(3-eps) * zp - kaH2(N, Nk_ini_to_hExit, Hk_exit, Hi) * z - 2*w(wmax, N, Nfeat_center, deltaNfeat) / np.sqrt(2*eps) * (Qsp+3*Qs)
    Qspp = -(3-eps) * Qsp - ( kaH2(N, Nk_ini_to_hExit, Hk_exit, Hi) + ms2H2(xi_ss, wmax, N, Nfeat_center, deltaNfeat) ) * Qs + 2*w(wmax, N, Nfeat_center, deltaNfeat) * np.sqrt(2*eps) * zp
    # return [zp, zpp]
    return [zp, zpp, Qsp, Qspp]

def cmbspectra(Ncmb_ini_to_hExit, Nk_ini_to_hExit, Nk_hExit_to_end, kcmb_MpcInv, Hi, eps, xi_ss, wmax, Nfeat_center, deltaNfeat):
    
    a_i_cmb     = a_ini_cmb(kcmb_MpcInv, Ncmb_i_to_exit, Hcmb)
    
    Hk_exit_arr  = np.zeros(num_k_cmb)
    logk         = np.zeros(num_k_cmb)
    
    logP_z_exit  = np.zeros(num_k_cmb)
    P_z_exit     = np.zeros(num_k_cmb)
    
    logP_Qs_exit = np.zeros(num_k_cmb)
    P_Qs_exit    = np.zeros(num_k_cmb)
    
    base_count = 10**4
    
    for i in range(len(Nk_ini_to_hExit)):
        
        numN    = int( (Nk_ini_to_hExit[i] + Nk_hExit_to_end) * base_count + 1 )
        Nk_time = np.linspace(0, Nk_ini_to_hExit[i] + Nk_hExit_to_end, numN)
        
        # if isinstance(Nk_hExit_to_end, np.ndarray) or isinstance(Nk_hExit_to_end, list):
            
        #     numN  = int( (Nk_ini_to_hExit + Nk_hExit_to_end[i]) * base_count + 1 )
        #     Nk_time = Nk_time_arr(Nk_ini_to_hExit, Nk_hExit_to_end[i], numN)
        
        # else:
            
        #     numN  = int( (Nk_ini_to_hExit + Nk_hExit_to_end) * base_count + 1 )
        #     Nk_time = Nk_time_arr(Nk_ini_to_hExit, Nk_hExit_to_end, numN)
        
        Hsoln          = odeint(H_eom, Hi, Nk_time, args=(eps,))[:,0]
        Hk_exit        = Hsoln[-1]
        Hk_exit_arr[i] = Hk_exit
        logk[i]        = logk_from_aH(a_i_cmb, Nk_ini_to_hExit[i], Hk_exit)
        
        
        zRe_i   =  1/2 * np.exp( 1/2*np.log(Hk_exit) - 3/2*np.log(a_i_cmb) - Nk_ini_to_hExit[i]/2 - 1/2*np.log(eps) - np.log(Hi) - np.log(Mp) ) * np.sin( np.exp( Nk_ini_to_hExit[i] * Hk_exit/Hi ) )
        zIm_i   = -1/2 * np.exp( 1/2*np.log(Hk_exit) - 3/2*np.log(a_i_cmb) - Nk_ini_to_hExit[i]/2 - 1/2*np.log(eps) - np.log(Hi) - np.log(Mp) ) * np.cos( np.exp( Nk_ini_to_hExit[i] * Hk_exit/Hi ) )
        
        zpRe_i  = -1/2 * np.exp( 3/2*np.log(Hk_exit) + Nk_ini_to_hExit[i]/2 - 3/2*np.log(a_i_cmb) - 1/2*np.log(eps) - 2*np.log(Hi) - np.log(Mp) ) * np.cos( np.exp( Nk_ini_to_hExit[i] * Hk_exit/Hi ) )
        zpIm_i  = -1/2 * np.exp( 3/2*np.log(Hk_exit) + Nk_ini_to_hExit[i]/2 - 3/2*np.log(a_i_cmb) - 1/2*np.log(eps) - 2*np.log(Hi) - np.log(Mp) ) * np.sin( np.exp( Nk_ini_to_hExit[i] * Hk_exit/Hi ) )
        
        QsRe_i  =  1/2 * np.exp( 1/2*np.log(Hk_exit) - 3/2*np.log(a_i_cmb) - Nk_ini_to_hExit[i]/2 - 1/2*np.log(eps) - np.log(Hi) - np.log(Mp) ) * np.sin( np.exp( Nk_ini_to_hExit[i] * Hk_exit/Hi ) )
        QsIm_i  = -1/2 * np.exp( 1/2*np.log(Hk_exit) - 3/2*np.log(a_i_cmb) - Nk_ini_to_hExit[i]/2 - 1/2*np.log(eps) - np.log(Hi) - np.log(Mp) ) * np.cos( np.exp( Nk_ini_to_hExit[i] * Hk_exit/Hi ) )
        
        QspRe_i = -1/2 * np.exp( 3/2*np.log(Hk_exit) + Nk_ini_to_hExit[i]/2 - 3/2*np.log(a_i_cmb) - 1/2*np.log(eps) - 2*np.log(Hi) - np.log(Mp) ) * np.cos( np.exp( Nk_ini_to_hExit[i] * Hk_exit/Hi ) )
        QspIm_i = -1/2 * np.exp( 3/2*np.log(Hk_exit) + Nk_ini_to_hExit[i]/2 - 3/2*np.log(a_i_cmb) - 1/2*np.log(eps) - 2*np.log(Hi) - np.log(Mp) ) * np.sin( np.exp( Nk_ini_to_hExit[i] * Hk_exit/Hi ) )
        
        # Single-field initial conditions:
        # ic_zrun_Re  = [zRe_i, zpRe_i]   # Re(zeta(0)) != 0, Re(zeta'(0)) != 0
        # ic_zrun_Im  = [zIm_i, zpIm_i]   # Im(zeta(0)) != 0, Im(zeta'(0)) != 0
        
        ic_zrun_Re  = [zRe_i, zpRe_i, 0, 0]   # Re(zeta(0)) != 0, Re(zeta'(0)) != 0, Re(Qs(0)) = 0, Re(Qs'(0)) = 0
        ic_zrun_Im  = [zIm_i, zpIm_i, 0, 0]   # Im(zeta(0)) != 0, Im(zeta'(0)) != 0, Im(Qs(0)) = 0, Im(Qs'(0)) = 0
        ic_Qsrun_Re = [0, 0, QsRe_i, QspRe_i] # Re(zeta(0)) = 0, Re(zeta'(0)) = 0, Re(Qs(0)) != 0, Re(Qs'(0)) != 0
        ic_Qsrun_Im = [0, 0, QsIm_i, QspIm_i] # Im(zeta(0)) = 0, Im(zeta'(0)) = 0, Im(Qs(0)) != 0, Im(Qs'(0)) != 0
        
        
        pertsoln_zrun_Re  = odeint(eoms, ic_zrun_Re, Nk_time, args=(Nk_ini_to_hExit[i], Hk_exit, Hi, eps, xi_ss, wmax, Nfeat_center, deltaNfeat))
        pertsoln_zrun_Im  = odeint(eoms, ic_zrun_Im, Nk_time, args=(Nk_ini_to_hExit[i], Hk_exit, Hi, eps, xi_ss, wmax, Nfeat_center, deltaNfeat))

        pertsoln_Qsrun_Re = odeint(eoms, ic_Qsrun_Re, Nk_time, args=(Nk_ini_to_hExit[i], Hk_exit, Hi, eps, xi_ss, wmax, Nfeat_center, deltaNfeat))
        pertsoln_Qsrun_Im = odeint(eoms, ic_Qsrun_Im, Nk_time, args=(Nk_ini_to_hExit[i], Hk_exit, Hi, eps, xi_ss, wmax, Nfeat_center, deltaNfeat))
        
        
        z_Re_zrun    = pertsoln_zrun_Re[:,0]
        Qs_Re_zrun   = pertsoln_zrun_Re[:,2]

        z_Im_zrun    = pertsoln_zrun_Im[:,0]
        Qs_Im_zrun   = pertsoln_zrun_Im[:,2]

        z_Re_Qsrun   = pertsoln_Qsrun_Re[:,0]
        Qs_Re_Qsrun  = pertsoln_Qsrun_Re[:,2]

        z_Im_Qsrun   = pertsoln_Qsrun_Im[:,0]
        Qs_Im_Qsrun  = pertsoln_Qsrun_Im[:,2]


        z2_zrun      = z_Re_zrun**2 + z_Im_zrun**2
        Qs2_zrun     = Qs_Re_zrun**2 + Qs_Im_zrun**2

        z2_Qsrun     = z_Re_Qsrun**2 + z_Im_Qsrun**2
        Qs2_Qsrun    = Qs_Re_Qsrun**2 + Qs_Im_Qsrun**2


        # z2           = z2_zrun #+ z2_Qsrun
        z2           = z2_zrun + z2_Qsrun
        Qs2          = Qs2_zrun + Qs2_Qsrun
        
        
        logP_z  = 3*logk[i] - np.log(2*np.pi**2) + np.log(z2)
        P_z     = np.exp(logP_z)
        logP_Qs = 3*logk[i] - np.log(2*np.pi**2) + np.log(Qs2)
        P_Qs    = np.exp(logP_Qs)
        
        # If choosing horizon exit value when evolving beyond horizon exit, use ind_Nk. Otherwise use freeze-out value at end of inflation
        # ind_Nk = find_nearest(Nk_time, Nk_ini_to_hExit[i])
        logP_z_exit[i]  = logP_z[-1]
        P_z_exit[i]     = P_z[-1]
        logP_Qs_exit[i] = logP_Qs[-1]
        P_Qs_exit[i]    = P_Qs[-1]
    
    # return [logk, Hk_exit_arr, P_z_exit, logP_z_exit] #, P_Qs_exit, logP_Qs_exit]
    return [logk, Hk_exit_arr, P_z_exit, logP_z_exit, P_Qs_exit, logP_Qs_exit]


#%% Compute CMB power spectra at horizon exit

Nk_i_to_exit    = deltaNk_cmb_list + Ncmb_i_to_exit
Nk_exit_to_stop = 0 # number of e-folds from horizon exit to end of evolving perturbations; use Nk_cmb_e2e_list to run until end of inflation
cmbsoln         = cmbspectra(Ncmb_i_to_exit, Nk_i_to_exit, Nk_exit_to_stop, kcmb, Hcmb_i, eH, xi, w0, Nfeat, deltaNfeat)

logk_arr        = cmbsoln[0]
Hk_exit_arr     = cmbsoln[1]
P_z_exit        = cmbsoln[2]
logP_z_exit     = cmbsoln[3]
# P_Qs_exit      = cmbsoln[4]
# logP_Qs_exit   = cmbsoln[5]


#%% Plot log(P_\zeta(k) / P_\zeta(k_*)) vs log(k/k_*)

ind_cmb_mode = int((num_k_cmb-1)/2) # middle of range of scales evaluated
logk_by_kstar = logk_arr - logk_arr[ ind_cmb_mode ]
logPzk_by_Pzkstar = logP_z_exit - logP_z_exit[ ind_cmb_mode ]

plt.xlabel(r'$\log[k/k_*]$')
plt.ylabel(r'$\log[P_\zeta(k) / P_\zeta(k_*)]$')
plt.plot( logk_by_kstar, logPzk_by_Pzkstar )


#%% Compute scalar spectral index n_s

logP_z_interpol = CubicSpline( logk_by_kstar, logPzk_by_Pzkstar )
ns = 1 + logP_z_interpol(0, 1)
print("ns = ", ns)


#%% Plot zeta power spectra vs horizon exit time

plt.xlabel(r'$N_{*,k}$')
plt.ylabel(r'$P_\zeta$')
plt.plot(-Nk_cmb_e2e_list, P_z_exit)


#%% Plot log(P_\zeta) vs log(k)

plt.xlabel(r'$\log[k]$')
plt.ylabel(r'$\log[P_\zeta]$')
plt.plot(logk_arr, logP_z_exit)


#%%

# =============================================================================
# BELOW IS THE SINGLE-k VERSION OF THE CODE
# Use if power spectrum for all time values is desired
# =============================================================================


#%% Define e-fold time array for mode k being evaluated

ind_cmb_mode = int((num_k_cmb-1)/2) # middle of range of scales evaluated
Nk_i_to_exit = Ncmb_i_to_exit + deltaNk_cmb_list[ind_cmb_mode] # use ind_cmb_mode if CMB pivot scale power spectrum is desired
Nk_exit_to_stop = Nk_i_to_exit # number of e-folds from horizon exit to end of evolving perturbations

numN  = int( (Nk_i_to_exit + Nk_exit_to_stop) * 10**4 + 1 ) # include + Ncmb_exit_to_end if evolving mode k to end of inflation
Nk_time = np.linspace(0, Nk_i_to_exit + Nk_exit_to_stop, numN) # include + Ncmb_exit_to_end in second arg if evolving mode k to end of inflation


#%% Compute k exiting horizon at Nk_i_to_exit + deltaNk_cmb_list[i]

# Use horizon exit condition, k = (aH)_exit

a_i_cmb = a_ini_cmb(kcmb, Ncmb_i_to_exit, Hcmb)

Hsoln          = odeint(H_eom, Hcmb_i, Nk_time, args=(eH,))[:,0]
Hk_exit        = Hsoln[-1]
logk           = logk_from_aH(a_i_cmb, Nk_i_to_exit, Hk_exit)


#%% Define initial conditions

# Specify nonzero initial condition values
# Note that different runs will choose different nonzero ICs:
#     z-run chooses z and zp nonvanishing (both Re and Im parts)
#     Qs-run chooses Qs and Qsp nonvanishing (both Re and Im parts)

zRe_i       =  1/2 * np.exp( 1/2*np.log(Hk_exit) - 3/2*np.log(a_i_cmb) - Nk_i_to_exit/2 - 1/2*np.log(eH) - np.log(Hcmb_i) - np.log(Mp) ) * np.sin( np.exp( Nk_i_to_exit * Hk_exit/Hcmb_i ) )
zIm_i       = -1/2 * np.exp( 1/2*np.log(Hk_exit) - 3/2*np.log(a_i_cmb) - Nk_i_to_exit/2 - 1/2*np.log(eH) - np.log(Hcmb_i) - np.log(Mp) ) * np.cos( np.exp( Nk_i_to_exit * Hk_exit/Hcmb_i ) )

zpRe_i      = -1/2 * np.exp( 3/2*np.log(Hk_exit) + Nk_i_to_exit/2 - 3/2*np.log(a_i_cmb) - 1/2*np.log(eH) - 2*np.log(Hcmb_i) - np.log(Mp) ) * np.cos( np.exp( Nk_i_to_exit * Hk_exit/Hcmb_i ) )
zpIm_i      = -1/2 * np.exp( 3/2*np.log(Hk_exit) + Nk_i_to_exit/2 - 3/2*np.log(a_i_cmb) - 1/2*np.log(eH) - 2*np.log(Hcmb_i) - np.log(Mp) ) * np.sin( np.exp( Nk_i_to_exit * Hk_exit/Hcmb_i ) )

QsRe_i      =  1/2 * np.exp( 1/2*np.log(Hk_exit) - 3/2*np.log(a_i_cmb) - Nk_i_to_exit/2 - 1/2*np.log(eH) - np.log(Hcmb_i) - np.log(Mp) ) * np.sin( np.exp( Nk_i_to_exit * Hk_exit/Hcmb_i ) )
QsIm_i      = -1/2 * np.exp( 1/2*np.log(Hk_exit) - 3/2*np.log(a_i_cmb) - Nk_i_to_exit/2 - 1/2*np.log(eH) - np.log(Hcmb_i) - np.log(Mp) ) * np.cos( np.exp( Nk_i_to_exit * Hk_exit/Hcmb_i ) )

QspRe_i     = -1/2 * np.exp( 3/2*np.log(Hk_exit) + Nk_i_to_exit/2 - 3/2*np.log(a_i_cmb) - 1/2*np.log(eH) - 2*np.log(Hcmb_i) - np.log(Mp) ) * np.cos( np.exp( Nk_i_to_exit * Hk_exit/Hcmb_i ) )
QspIm_i     = -1/2 * np.exp( 3/2*np.log(Hk_exit) + Nk_i_to_exit/2 - 3/2*np.log(a_i_cmb) - 1/2*np.log(eH) - 2*np.log(Hcmb_i) - np.log(Mp) ) * np.sin( np.exp( Nk_i_to_exit * Hk_exit/Hcmb_i ) )

# Single-field initial conditions:
# ic_zrun_Re  = [zRe_i, zpRe_i]   # Re(zeta(0)) != 0, Re(zeta'(0)) != 0
# ic_zrun_Im  = [zIm_i, zpIm_i]   # Im(zeta(0)) != 0, Im(zeta'(0)) != 0

ic_zrun_Re  = [zRe_i, zpRe_i, 0, 0]   # Re(zeta(0)) != 0, Re(zeta'(0)) != 0, Re(Qs(0)) = 0, Re(Qs'(0)) = 0
ic_zrun_Im  = [zIm_i, zpIm_i, 0, 0]   # Im(zeta(0)) != 0, Im(zeta'(0)) != 0, Im(Qs(0)) = 0, Im(Qs'(0)) = 0
ic_Qsrun_Re = [0, 0, QsRe_i, QspRe_i] # Re(zeta(0)) = 0, Re(zeta'(0)) = 0, Re(Qs(0)) != 0, Re(Qs'(0)) != 0
ic_Qsrun_Im = [0, 0, QsIm_i, QspIm_i] # Im(zeta(0)) = 0, Im(zeta'(0)) = 0, Im(Qs(0)) != 0, Im(Qs'(0)) != 0


#%% Solve equations of motion

pertsoln_zrun_Re  = odeint(eoms, ic_zrun_Re, Nk_time, args=(Nk_i_to_exit, Hk_exit, Hcmb_i, eH, xi, w0, Nfeat, deltaNfeat))
pertsoln_zrun_Im  = odeint(eoms, ic_zrun_Im, Nk_time, args=(Nk_i_to_exit, Hk_exit, Hcmb_i, eH, xi, w0, Nfeat, deltaNfeat))

pertsoln_Qsrun_Re = odeint(eoms, ic_Qsrun_Re, Nk_time, args=(Nk_i_to_exit, Hk_exit, Hcmb_i, eH, xi, w0, Nfeat, deltaNfeat))
pertsoln_Qsrun_Im = odeint(eoms, ic_Qsrun_Im, Nk_time, args=(Nk_i_to_exit, Hk_exit, Hcmb_i, eH, xi, w0, Nfeat, deltaNfeat))


#%% Extract solutions

z_Re_zrun    = pertsoln_zrun_Re[:,0]
zp_Re_zrun   = pertsoln_zrun_Re[:,1]
Qs_Re_zrun   = pertsoln_zrun_Re[:,2]
Qsp_Re_zrun  = pertsoln_zrun_Re[:,3]

z_Im_zrun    = pertsoln_zrun_Im[:,0]
zp_Im_zrun   = pertsoln_zrun_Im[:,1]
Qs_Im_zrun   = pertsoln_zrun_Im[:,2]
Qsp_Im_zrun  = pertsoln_zrun_Im[:,3]

z_Re_Qsrun   = pertsoln_Qsrun_Re[:,0]
zp_Re_Qsrun  = pertsoln_Qsrun_Re[:,1]
Qs_Re_Qsrun  = pertsoln_Qsrun_Re[:,2]
Qsp_Re_Qsrun = pertsoln_Qsrun_Re[:,3]

z_Im_Qsrun   = pertsoln_Qsrun_Im[:,0]
zp_Im_Qsrun  = pertsoln_Qsrun_Im[:,1]
Qs_Im_Qsrun  = pertsoln_Qsrun_Im[:,2]
Qsp_Im_Qsrun = pertsoln_Qsrun_Im[:,3]


z2_zrun      = z_Re_zrun**2 + z_Im_zrun**2
Qs2_zrun     = Qs_Re_zrun**2 + Qs_Im_zrun**2

z2_Qsrun     = z_Re_Qsrun**2 + z_Im_Qsrun**2
Qs2_Qsrun    = Qs_Re_Qsrun**2 + Qs_Im_Qsrun**2


# z2           = z2_zrun #+ z2_Qsrun
z2           = z2_zrun + z2_Qsrun
Qs2          = Qs2_zrun + Qs2_Qsrun


#%% Compute power spectra

logP_z  = 3*logk - np.log(2*np.pi**2) + np.log(z2)
P_z     = np.exp(logP_z)
logP_Qs = 3*logk - np.log(2*np.pi**2) + np.log(Qs2)
P_Qs    = np.exp(logP_Qs)


#%% Plot log(P_zeta) vs N

plt.xlabel('N')
plt.ylabel(r'$\log(P_\zeta)$')
plt.plot(Nk_time, logP_z)


#%% Plot zeta power spectra at early times

end = int((numN-1)*0.01)

plt.xlabel('N')
plt.ylabel(r'$P_\zeta$')
plt.plot(Nk_time[0:end], P_z[0:end])


#%% Plot zeta power spectra at late times

start = int((numN-1)*0.5) # counting backwards from the end

plt.xlabel('N')
plt.ylabel(r'$P_\zeta$')
plt.plot(Nk_time[-start:-1], P_z[-start:-1])


#%% Plot Qs power spectra at late times

# start = int((numN-1)*0.1) # counting backwards from the end

# plt.xlabel('N')
# plt.ylabel(r'$P_{Q_s}$')
# plt.plot(Nk_time[-start:-1], P_Qs[-start:-1])










