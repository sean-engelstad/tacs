import numpy as np
import os
from tacs import pytacs, TACS, elements, constitutive, functions, problems
from pytacs_analysis_base_test import PyTACSTestCase

'''
6 noded beam model 1 meter long in x direction.
The cross-sectional properties of the beam are as follows:
    A = 0.1
    Iz = 0.2
    Iy = 0.3
    J = 0.4
    Iyz = -0.1
Because Iyz =/= 0.0, we expect some coupling to show up in y and z bending. 
We apply apply various tip loads test KSDisplacement, StructuralMass, and Compliance functions and sensitivities.
'''

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_file = os.path.join(base_dir, "./input_files/beam_model.bdf")

FUNC_REFS = {'z-shear_compliance': 1.3200259999099195,
             'z-shear_mass': 0.27,
             'z-shear_x_disp': 0.0, 'z-shear_y_disp': -0.29391826170251795, 'z-shear_z_disp': 12.141597029011256,

             'y-shear_compliance': 1.9800259997664378,
             'y-shear_mass': 0.27,
             'y-shear_x_disp': 0.0, 'y-shear_y_disp': 18.327400292118664, 'y-shear_z_disp': -0.2939182616984911,

             'x-axial_compliance': 10.000000000000027,
             'x-axial_mass': 0.27,
             'x-axial_x_disp': 95.543244182597, 'x-axial_y_disp': 0.0, 'x-axial_z_disp': 0.0,

             'x-torsion_compliance': 6.499999999999995,
             'x-torsion_mass': 0.27,
             'x-torsion_x_disp': 0.0, 'x-torsion_y_disp': 0.0, 'x-torsion_z_disp': 0.0}

ksweight = 10.0

class ProblemTest(PyTACSTestCase.PyTACSTest):
    N_PROCS = 2  # this is how many MPI processes to use for this TestCase.
    def setup_pytacs(self, comm, dtype):
        """
        Setup mesh and pytacs object for problem we will be testing.
        """

        # Overwrite default check values
        if dtype == complex:
            self.rtol = 1e-6
            self.atol = 1e-6
            self.dh = 1e-50
        else:
            self.rtol = 2e-1
            self.atol = 1e-3
            self.dh = 1e-6

        # Instantiate FEA Assembler
        struct_options = {}

        fea_assembler = pytacs.pyTACS(bdf_file, comm, options=struct_options)

        # Set up constitutive objects and elements
        fea_assembler.initialize()

        return fea_assembler

    def setup_tacs_vecs(self, fea_assembler, dv_pert_vec, xpts_pert_vec):
        """
        Setup user-defined vectors for analysis and fd/cs sensitivity verification
        """
        # Create temporary dv vec for doing fd/cs
        dv_pert_vec[:] = 1.0

        # Define perturbation array that moves all nodes on shell
        xpts = fea_assembler.getOrigNodes()
        xpts_pert_vec[:] = xpts

        return

    def setup_funcs(self, fea_assembler, problems):
        """
        Create a list of functions to be tested and their reference values for the problem
        """
        # Add Functions
        for problem in problems:
            problem.addFunction('mass', functions.StructuralMass)
            problem.addFunction('compliance', functions.Compliance)
            problem.addFunction('x_disp', functions.KSDisplacement,
                                ksWeight=ksweight, direction=[10.0, 0.0, 0.0])
            problem.addFunction('y_disp', functions.KSDisplacement,
                                ksWeight=ksweight, direction=[0.0, 10.0, 0.0])
            problem.addFunction('z_disp', functions.KSDisplacement,
                                ksWeight=ksweight, direction=[0.0, 0.0, 10.0])
        func_list = ['mass', 'compliance', 'x_disp', 'y_disp', 'z_disp']
        return func_list, FUNC_REFS

    def setup_tacs_problems(self, fea_assembler):
        """
        Setup pytacs object for problems we will be testing.
        """
        # Read in forces from BDF and create tacs struct problems
        tacs_probs = fea_assembler.createTACSProbsFromBDF()
        # Convert from dict to list
        tacs_probs = tacs_probs.values()
        # Set convergence to be tight for test
        for problem in tacs_probs:
            problem.setOption('L2Convergence', 1e-20)
            problem.setOption('L2ConvergenceRel', 1e-20)

        return tacs_probs