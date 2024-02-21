import numpy as np
from tacs import buckling_surrogate
import matplotlib.pyplot as plt
import niceplots, pandas, os

# vary aspect ratio and for D* = 1
# show that the equations match the affine transformation from 
# "Generic Buckling Curves For Specially Orthotropic Rectangular Plates" by Brunelle

AR_list = []
kx0_FEA_list = [] # FEA
kx0_CF_list = [] # closed-form

# clear the csv file
csv_file = "axial-SS1.csv"
if os.path.exists(csv_file):
    os.remove(csv_file)

ct = 0
for AR in np.linspace(0.5, 5, 40):

    AR_list += [AR]

    # since material here is isotropic, D11=D22
    # and a0/b0 = a/b = AR
    affine_AR = AR

    flat_plate = buckling_surrogate.FlatPlateAnalysis(
        bdf_file="plate.bdf",
        a=AR,
        b=1.0,
        h=0.01, # very slender => so near thin plate limit
        E11=70e9,
        nu12=0.33,
        E22=None,  # set to None if isotropic
        G12=None,  # set to None if isotropic
    )

    # select number of elements
    if AR > 1.0:
        nx = np.min([int(AR * 30), 100])
        ny = 30
    else:  # AR < 1.0
        ny = np.min([int(AR * 30), 100])
        nx = 30

    flat_plate.generate_bdf(
        nx=nx,
        ny=ny,
        exx=flat_plate.affine_exx,
        eyy=0.0,
        exy=0.0,
        clamped=False,
    )

    # avg_stresses = flat_plate.run_static_analysis(write_soln=True)

    tacs_eigvals = flat_plate.run_buckling_analysis(sigma=5.0, num_eig=6, write_soln=True)

    # expect to get ~4.5
    # since k0-2D* = (m a_0/b_0)^2 + (b_0/a_0/m)^2
    # in affine space and D*=1 and k0-2D* = 2.5 in Brunelle paper (buckling-analysis section)
    # "Generic Buckling Curves For Specially Orthotropic Rectangular Plates"
    # but only works in thin plate limit (very thin)

    # the non-dimensional buckling coefficient
    kx0_FEA_list += [tacs_eigvals[0]]

    # compute kx0 from the closed-form solution of the affine transform paper
    # loop over each mode and find the min non-dimensional buckling coefficient
    kx0_modes = []
    for m in range(1,6):
        kx0_modes += [2.0*flat_plate.Dstar + (m / affine_AR)**2 + (affine_AR/m)**2]
    kx0_exact = min(kx0_modes)
    kx0_CF_list += [kx0_exact]

    data_dict = {
        "AR" : AR_list[-1:],
        "kx0_FEA" : kx0_FEA_list[-1:],
        "kx0_CF" : kx0_CF_list[-1:],
    }

    # write the data to a file
    df = pandas.DataFrame(data_dict)
    if ct == 0:
        df.to_csv(csv_file, mode="w", index=False)
    else:
        df.to_csv(csv_file, mode="a", index=False, header=False)
    ct += 1
    
# plot the Closed-Form versus the FEA for affine equations
plt.style.use(niceplots.get_style())
plt.figure('affine')
plt.plot(AR_list, kx0_CF_list, label="closed-form")
plt.plot(AR_list, kx0_FEA_list, "o", label="FEA")
plt.xlabel(r"$a_0/b_0$")
plt.ylabel(r"$k_{x_0}$")
plt.legend()
plt.savefig("affine-AR-kx0.png", dpi=400)