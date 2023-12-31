"""
Sean Engelstad, January 2024
GT SMDO Lab, Dr. Graeme Kennedy
Caps to TACS example
"""

from tacs import caps2tacs, pyTACS
from mpi4py import MPI
import os
from pprint import pprint

# run a steady elastic structural analysis in TACS using the tacsAIM wrapper caps2tacs submodule
# -------------------------------------------------------------------------------------------------
# 1: build the tacs aim, egads aim wrapper classes

comm = MPI.COMM_WORLD

print(f"proc on rank {comm.rank}")

tacs_model = caps2tacs.TacsModel.build(
    csm_file="plate.csm", comm=comm, active_procs=[0]
)
tacs_model.mesh_aim.set_mesh(
    edge_pt_min=15,
    edge_pt_max=20,
    global_mesh_size=0.25,
    max_surf_offset=0.01,
    max_dihedral_angle=15,
).register_to(tacs_model)
tacs_aim = tacs_model.tacs_aim

aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)
caps2tacs.ThicknessVariable(
    caps_group="plate", value=0.05, material=aluminum
).register_to(tacs_model)

# add constraints and loads
caps2tacs.PinConstraint("nn", dof_constraint=123).register_to(tacs_model)
caps2tacs.PinConstraint("np", dof_constraint=13).register_to(tacs_model)
caps2tacs.PinConstraint("pn", dof_constraint=23).register_to(tacs_model)
caps2tacs.PinConstraint("pp", dof_constraint=3).register_to(tacs_model)

caps2tacs.GridForce("Xn", direction=[1.0, 0.0, 0.0], magnitude=100.0).register_to(
    tacs_model
)
caps2tacs.GridForce("Xp", direction=[-1.0, 0.0, 0.0], magnitude=100.0).register_to(
    tacs_model
)
caps2tacs.GridForce("Yn", direction=[0.0, +1.0, 0.0], magnitude=100.0).register_to(
    tacs_model
)
caps2tacs.GridForce("Yp", direction=[0.0, -1.0, 0.0], magnitude=100.0).register_to(
    tacs_model
)

# run the pre analysis to build tacs input files
# alternative is to call tacs_aim.setup_aim().pre_analysis() with tacs_aim = tacs_model.tacs_aim
tacs_model.setup(include_aim=True)
tacs_model.pre_analysis()

comm.Barrier()

# ----------------------------------------------------------------------------------
# 2. Run the TACS steady elastic structural analysis, forward + adjoint

# solve the buckling analysis
# Instantiate FEAAssembler
bdfFile = tacs_aim.root_dat_file
FEAAssembler = pyTACS(bdfFile, comm=comm)
# Set up constitutive objects and elements
FEAAssembler.initialize()

# Setup static problem
if FEAAssembler.bdfInfo.is_xrefed is False:
    FEAAssembler.bdfInfo.cross_reference()
    FEAAssembler.bdfInfo.is_xrefed = True
bucklingProb = FEAAssembler.createBucklingProblem(name="buckle", sigma=10.0, numEigs=5)
bucklingProb.setOption("printLevel", 2)
# Add Loads
bucklingProb.addLoadFromBDF(loadID=1)

# solve and evaluate functions/sensitivities
funcs = {}
funcsSens = {}
bucklingProb.solve()
bucklingProb.evalFunctions(funcs)
bucklingProb.evalFunctionsSens(funcsSens)
bucklingProb.writeSolution(outputDir=os.path.dirname(__file__))


# if comm.rank == 0:
#    pprint(funcs)
#    pprint(funcsSens)
