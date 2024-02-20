"""
Sean Engelstad
Feb 2024, GT SMDO Lab
Goal is to generate data for pure uniaxial compression failure in the x-direction
For now just simply supported only..
"""

from _generate_plate import generate_plate
from _static_analysis import run_static_analysis
from _buckling_analysis import run_buckling_analysis
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import niceplots

"""
Let's consider the following parameters in exploring the data..
size: Lx, Ly, h (thickness)
material: E, nu, or D = E*h^3/12/(1-nu)
others: aspect ratio Lx/Ly
    slenderness ratio Lx/h, or Ly/h related to shear strain energy
    compute lambda * Nx = Nxcrit as the output (Nxcrit)
"""

# make a data dict for storing pandas data (later converted to dataframe)
keys = ["Dstar", "a0/b0", "b/h", "kx_0"]
data_dict = {key: [] for key in keys}

# make a folder for the pandas data
cpath = os.path.dirname(__file__)
data_folder = os.path.join(cpath, "data")
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

# TODO : change this to a Monte carlo simulation over uniform scales or uniform log scales
# TODO : use the affine transformation of 1_run_analysis.py to make lambda values more consistent.

# arrays to check the values of 

accepted_ct = 0
ct = 0
while accepted_ct < 10: # until has generated this many samples
    # randomly generate the material constants
    log_Emax = np.log10(200e9)
    log_E11 = np.random.uniform(9, log_Emax)
    E11 = 10**log_E11
    log_E22 = np.random.uniform(9, log_Emax)
    E22 = 10**log_E22
    log_G12 = np.random.uniform(9, log_Emax)
    G12 = 10**log_G12
    nu12 = np.random.uniform(-0.95, 0.45)

    # randomly generate the plate sizes
    log_a = np.random.uniform(-1, 1)
    a = 10**log_a
    log_b = np.random.uniform(-1,1)
    b = 10**log_b
    log_h = np.random.uniform(-3,-2)
    h = 10**log_h

    # get main model parameters
    Dstar = nu12 * np.sqrt(E22/E11) + 2 * G12 * (1 - nu12**2 * E22/E11) / np.sqrt(E11 * E22)
    a0_b0 = (E22/E11)**0.25 * a/b
    log_a0_b0 = np.log10(a0_b0)
    slenderR = b/h
    a_b = a/b
    AR = a_b
    log_slender = np.log10(slenderR)

    # check the model parameter ranges are reasonable
    valid_Dstar = 0 <= Dstar <= 1.0
    valid_a_b = 0.05 <= a_b <= 20.0
    valid_a0_b0 = 0.05 <= a0_b0 <= 20.0
    valid_bh = 5 <= slenderR <= 200

    # skip this random model if model parameters are outside ranges
    ct += 1
    if valid_Dstar and valid_a0_b0 and valid_a_b and valid_bh:
        pass
    else:
        _print_out_of_bounds = False
        if _print_out_of_bounds:
            if not(valid_Dstar):
                print(f"invalid Dstar = {Dstar}")
            if not(valid_a_b):
                print(f"invalid a/b = {a_b}")
            if not(valid_a0_b0):
                print(f"invalid a0/b0 = {a0_b0}")
            if not(valid_bh):
                print(f"invalid b/h = {slenderR}")
        continue

    # if model parameters were in range then we can now run the buckling analysis

    # compute parameters for the affine bending problem
    nu21 = E22 / E11 * nu12
    D11 = E11 * h**3 / 12.0 / (1 - nu12 * nu21)
    D22 = E22 * h**3 / 12.0 / (1 - nu12 * nu21)

    # select number of elements
    if AR > 1.0:
        nx = np.min([AR*20, 50])
        ny = 20
    else: # AR < 1.0
        ny = np.min([AR*20, 50])
        nx = 20

    # choose uniaxial compressive strain so that output lambda = k_x0 of buckling analysis
    # see 1_run_analysis and affine CPT derivations
    exx_T = np.pi**2 * np.sqrt(D11 * D22) / b**2 / h / E11

    _run_buckling = True

    if _run_buckling:

        generate_plate(Lx=a, Ly=b, nx=30, ny=30, exx=exx_T, eyy=0.0, exy=0.0, clamped=False)
        # run_static_analysis(thickness=h, E=E, nu=nu, write_soln=True)

        tacs_eigvals = run_buckling_analysis(
            thickness=h,
            E11=E11,
            nu12=nu12,
            E22=E22,
            G12=G12,
            sigma=10.0,
            num_eig=12,
            write_soln=True,
        )
        
        kx_0 = tacs_eigvals[0]

    else: # just do a model parameter check
        kx_0 = 1.0 # for model parameter check
    
    if kx_0 > 0:
        accepted_ct += 1

        # log the model parameters
        data_dict["Dstar"] += [Dstar]
        data_dict["a0/b0"] += [a0_b0]
        data_dict["b/h"] += [slenderR]
        data_dict["kx_0"] += [kx_0]

    else:
        continue

# report the percentage of good models out of the Monte Carlo simulation
frac_good_models = accepted_ct * 1.0 / ct
print(f"fraction of good models: {accepted_ct} / {ct} = {frac_good_models}")

# check the model parameter ranges were covered
_check_params = False
plt.style.use(niceplots.get_style())
if _check_params:

    plt.figure("Dstar-AR")
    plt.plot(data_dict["Dstar"], data_dict["a0/b0"], "ko")
    plt.yscale("log")
    plt.xlabel("Dstar")
    plt.ylabel("AR")
    plt.show()
    plt.close("Dstar-AR")

    plt.figure("Dstar-slenderR")
    plt.plot(data_dict["Dstar"], data_dict["b/h"], "ko")
    plt.yscale("log")
    plt.xlabel("Dstar")
    plt.ylabel("slenderR")
    plt.show()
    plt.close("Dstar-slenderR")

# plot aspect ratio versus kx0
plt.plot(data_dict["Dstar"], data_dict["kx_0"])
plt.savefig("data/Dstar-kx0.png", dpi=400)


# convert to a pandas dataframe and save it in the data folder
df = pd.DataFrame(data_dict)
csv_file = os.path.join(data_folder, "Nxcrit.csv")
df.to_csv(csv_file)
