/*
=====================================================================================================
Blade-Stiffened Shell Constitutive Model using Gaussian Process Machine Learning
Buckling Constraints
=====================================================================================================
@File    :   TACSGPBladeStiffenedShellConstutive.cpp
@Date    :   2024/04/24
@Author  :   Sean Phillip Engelstad, Alasdair Christian Gray
@Description : Constitutive model for a blade-stiffened shell. Based on the FSDT
blade models adopted by Alasdair Christian Gray in the
TACSBladeStiffenedShellConstutitutive.h class and the original one by Graeme
Kennedy. Gaussian Processes for Machine Learning are used for the buckling
constraints of the stiffened panels.
*/

// =============================================================================
// Standard Library Includes
// =============================================================================

// =============================================================================
// Extension Includes
// =============================================================================
#include "TACSGPBladeStiffenedShellConstitutive.h"

#include "TACSMaterialProperties.h"
#include "TACSShellConstitutive.h"

const char* TACSGPBladeStiffenedShellConstitutive::constName =
    "TACSGPBladeStiffenedShellConstitutive";

// ==============================================================================
// Constructor
// ==============================================================================

TACSGPBladeStiffenedShellConstitutive::TACSGPBladeStiffenedShellConstitutive(
    TACSOrthotropicPly* panelPly, TACSOrthotropicPly* stiffenerPly,
    TacsScalar kcorr, TacsScalar panelLength, int panelLengthNum,
    TacsScalar stiffenerPitch, int stiffenerPitchNum, TacsScalar panelThick,
    int panelThickNum, int numPanelPlies, TacsScalar panelPlyAngles[],
    TacsScalar panelPlyFracs[], int panelPlyFracNums[],
    TacsScalar stiffenerHeight, int stiffenerHeightNum,
    TacsScalar stiffenerThick, int stiffenerThickNum, int numStiffenerPlies,
    TacsScalar stiffenerPlyAngles[], TacsScalar stiffenerPlyFracs[],
    int stiffenerPlyFracNums[], TacsScalar panelWidth, int panelWidthNum,
    TacsScalar flangeFraction, AxialGaussianProcessModel* axialGP,
    ShearGaussianProcessModel* shearGP,
    CripplingGaussianProcessModel* cripplingGP)
    : TACSBladeStiffenedShellConstitutive(
          panelPly, stiffenerPly, kcorr, panelLength, panelLengthNum,
          stiffenerPitch, stiffenerPitchNum, panelThick, panelThickNum,
          numPanelPlies, panelPlyAngles, panelPlyFracs, panelPlyFracNums,
          stiffenerHeight, stiffenerHeightNum, stiffenerThick,
          stiffenerThickNum, numStiffenerPlies, stiffenerPlyAngles,
          stiffenerPlyFracs, stiffenerPlyFracNums, flangeFraction) {
  // DVs section, only one new DV - panelWidth
  // --- Panel width values ---
  this->panelWidth = panelWidth;
  this->panelWidthNum = panelWidthNum;
  this->panelWidthLocalNum = -1;
  if (panelWidthNum >= 0) {
    this->panelWidthLocalNum = this->numDesignVars;
    this->numDesignVars++;
    this->numGeneralDV++;
  }
  this->panelWidthLowerBound = 0.000;
  this->panelWidthUpperBound = 1e20;

  // set Gaussian process models in
  this->axialGP = axialGP;
  // if (this->axialGP) {
  //   this->axialGP->incref();
  // }

  this->shearGP = shearGP;
  this->cripplingGP = cripplingGP;
}

// ==============================================================================
// Destructor
// ==============================================================================
TACSGPBladeStiffenedShellConstitutive::
    ~TACSGPBladeStiffenedShellConstitutive() {
  printf("C++ GPConst destructor: start\n");

  // call superclass destructor
  TACSBladeStiffenedShellConstitutive::~TACSBladeStiffenedShellConstitutive();
  printf("C++ GPConst destructor: done with subclass destructor\n");

  // destroy the gaussian process model objects if they exist
  if (this->axialGP) {
    printf("C++ GPconst destructor: attempt destroy axialGP pointer ",
           this->axialGP);
    delete this->axialGP;
    this->axialGP = nullptr;
  } else {
    printf("C++ GPConst Destructor: has nullptr for axialGP\n");
  }
  printf("C++ GPConst destructor: destroyed axialGP\n");

  if (this->shearGP) {
    printf("C++ GPconst destructor: attempt destroy shearGP pointer ",
           this->shearGP);
    delete this->shearGP;
    this->shearGP = nullptr;
  } else {
    printf("C++ GPConst Destructor: has nullptr for shearGP\n");
  }
  printf("C++ GPConst destructor: destroyed shearGP\n");

  if (this->cripplingGP) {
    printf("C++ GPconst destructor: attempt destroy cripplingGP pointer ",
           this->cripplingGP);
    delete this->cripplingGP;
    this->cripplingGP = nullptr;
  } else {
    printf("C++ GPConst Destructor: has nullptr for cripplingGP\n");
  }
  printf("C++ GPConst destructor: destroyed cripplingGP\n");
  printf("C++ GPConst destructor: exit destructor\n");
}

// ==============================================================================
// Override Failure constraint and sensitivities
// ==============================================================================

// Compute the failure values for each failure mode of the stiffened panel
TacsScalar TACSGPBladeStiffenedShellConstitutive::computeFailureValues(
    const TacsScalar e[], TacsScalar fails[]) {
  // --- #0 - Panel material failure ---
  fails[0] = this->computePanelFailure(e);

  // --- #1 - Stiffener material failure ---
  TacsScalar stiffenerStrain[TACSBeamConstitutive::NUM_STRESSES];
  this->transformStrain(e, stiffenerStrain);
  fails[1] = this->computeStiffenerFailure(stiffenerStrain);

  // -- prelim to buckling constraints --
  // compute the A,B,D matrices of the panel
  TacsScalar panelStiffness[NUM_TANGENT_STIFFNESS_ENTRIES],
      panelStress[NUM_STRESSES];
  this->computePanelStiffness(panelStiffness);
  const TacsScalar *Ap, *Dp;
  this->extractTangentStiffness(panelStiffness, &Ap, NULL, &Dp, NULL, NULL);
  this->computePanelStress(e, panelStress);

  // Compute panel dimensions, material props and non-dimensional parameters
  TacsScalar N1CritGlobal, N12CritGlobal;
  TacsScalar D11p = Dp[0], D12p = Dp[1], D22p = Dp[3], D66p = Dp[5];
  TacsScalar A11p = Ap[0], A66p = Ap[5];
  TacsScalar delta, rho0Panel, xiPanel, gamma, a, b, zetaPanel;
  a = this->panelLength;
  b = this->panelWidth;
  delta = computeStiffenerAreaRatio();
  rho0Panel = computeAffineAspectRatio(D11p, D22p, a, b);
  xiPanel = computeGeneralizedRigidity(D11p, D22p, D12p, D66p);
  gamma = computeStiffenerStiffnessRatio(D11p);
  zetaPanel = computeTransverseShearParameter(A66p, A11p, b, this->panelThick);

  // --- #2 - Global panel buckling ---
  // compute the global critical loads
  N1CritGlobal = computeCriticalGlobalAxialLoad(D11p, D22p, b, delta, rho0Panel,
                                                xiPanel, gamma, zetaPanel);
  // TODO : add other inputs like rho0 here and ML buckling constraints
  N12CritGlobal = computeCriticalGlobalShearLoad(D11p, D22p, b, xiPanel,
                                                 rho0Panel, gamma, zetaPanel);

  // combined failure criterion for axial + shear global mode buckling
  // panelStress[0] is the panel in-plane load Nx, panelStress[2] is panel shear
  // in-plane load Nxy NOTE : not using smeared properties here for the local
  // panel stress (closed-form instead for stiffener properties)
  fails[2] = this->bucklingEnvelope(-panelStress[0], N1CritGlobal,
                                    panelStress[2], N12CritGlobal);

  // --- #3 - Local panel buckling ---
  TacsScalar N1CritLocal, N12CritLocal;

  // compute the local critical loads
  N1CritLocal =
      computeCriticalLocalAxialLoad(D11p, D22p, rho0Panel, xiPanel, zetaPanel);
  // TODO : add other inputs like rho0 here and ML buckling constraints
  N12CritLocal =
      computeCriticalLocalShearLoad(D11p, D22p, xiPanel, rho0Panel, zetaPanel);

  // stress[0] is the panel in-plane load Nx, stress[2] is panel shear in-plane
  // load Nxy
  fails[3] = this->bucklingEnvelope(-panelStress[0], N1CritLocal,
                                    panelStress[2], N12CritLocal);

  // --- #4 - Stiffener crippling ---
  // compute the A,B,D matrices of the stiffener
  TacsScalar stiffenerStiffness[NUM_TANGENT_STIFFNESS_ENTRIES],
      stiffenerStress[NUM_STRESSES];
  this->computeStiffenerStiffness(stiffenerStiffness);
  const TacsScalar *As, *Ds;
  this->extractTangentStiffness(stiffenerStiffness, &As, NULL, &Ds, NULL, NULL);
  this->computeStiffenerStress(stiffenerStrain, stiffenerStress);

  // compute stiffener non-dimensional parameters
  TacsScalar D11s = Ds[0], D12s = Ds[1], D22s = Ds[3], D66s = Ds[5];
  TacsScalar A11s = As[0], A66s = As[5];
  TacsScalar rho0Stiff, xiStiff, bStiff, hStiff, genPoiss, zetaStiff;
  bStiff = this->stiffenerHeight;
  hStiff = this->stiffenerThick;
  rho0Stiff = computeAffineAspectRatio(D11s, D22s, a, bStiff);
  xiStiff = computeGeneralizedRigidity(D11s, D22s, D12s, D66s);
  genPoiss = computeGeneralizedPoissonsRatio(D12s, D66s);
  zetaStiff = computeTransverseShearParameter(A66s, A11s, bStiff, hStiff);

  // Compute panel dimensions, material props and non-dimensional parameters
  TacsScalar N1, N1CritCrippling;
  N1CritCrippling = computeStiffenerCripplingLoad(
      D11s, D22s, xiStiff, rho0Stiff, genPoiss, zetaStiff);
  N1 = -stiffenerStress[0];
  fails[4] = N1 / N1CritCrippling;
  // --- End of computeFailuresValues subsections ---

  // aggregate the failure across all 5 failures modes (0-4)
  return ksAggregation(fails, this->NUM_FAILURES, this->ksWeight);
}

// Evaluate the derivative of the failure criteria w.r.t. the strain
TacsScalar TACSGPBladeStiffenedShellConstitutive::evalFailureStrainSens(
    int elemIndex, const double pt[], const TacsScalar X[],
    const TacsScalar e[], TacsScalar sens[]) {
  // TODO : need to complete this function
  memset(sens, 0, this->NUM_STRESSES * sizeof(TacsScalar));

  // --- #0 - Panel material failure ---
  TacsScalar fails[this->NUM_FAILURES], dKSdf[this->NUM_FAILURES];
  TacsScalar panelFailSens[this->NUM_STRESSES];
  fails[0] = this->evalPanelFailureStrainSens(e, panelFailSens);

  // --- #1 - Stiffener material failure ---
  TacsScalar stiffenerStrain[TACSBeamConstitutive::NUM_STRESSES],
      stiffenerStrainSens[TACSBeamConstitutive::NUM_STRESSES],
      stiffenerFailSens[this->NUM_STRESSES];
  this->transformStrain(e, stiffenerStrain);
  fails[1] = this->evalStiffenerFailureStrainSens(stiffenerStrain,
                                                  stiffenerStrainSens);
  this->transformStrainSens(stiffenerStrainSens, stiffenerFailSens);

  // -- prelim to buckling constraints --
  // compute the A,B,D matrices of the panel
  TacsScalar panelStiffness[NUM_TANGENT_STIFFNESS_ENTRIES],
      panelStress[NUM_STRESSES];
  this->computePanelStiffness(panelStiffness);
  const TacsScalar *Ap, *Dp;
  this->extractTangentStiffness(panelStiffness, &Ap, NULL, &Dp, NULL, NULL);
  this->computePanelStress(e, panelStress);

  // Compute panel dimensions, material props and non-dimensional parameters
  TacsScalar N1CritGlobal, N12CritGlobal;
  TacsScalar D11p = Dp[0], D12p = Dp[1], D22p = Dp[3], D66p = Dp[5];
  TacsScalar A11p = Ap[0], A66p = Ap[5];
  TacsScalar delta, rho0Panel, xiPanel, gamma, a, b, zetaPanel;
  a = this->panelLength;
  b = this->panelWidth;
  delta = computeStiffenerAreaRatio();
  rho0Panel = computeAffineAspectRatio(D11p, D22p, a, b);
  xiPanel = computeGeneralizedRigidity(D11p, D22p, D12p, D66p);
  gamma = computeStiffenerStiffnessRatio(D11p);
  zetaPanel = computeTransverseShearParameter(A66p, A11p, b, this->panelThick);

  // --- #2 - Global panel buckling ---
  // compute the global critical loads
  N1CritGlobal = computeCriticalGlobalAxialLoad(D11p, D22p, b, delta, rho0Panel,
                                                xiPanel, gamma, zetaPanel);
  // TODO : add other inputs like rho0 here and ML buckling constraints
  N12CritGlobal = computeCriticalGlobalShearLoad(D11p, D22p, b, xiPanel,
                                                 rho0Panel, gamma, zetaPanel);

  // combined failure criterion for axial + shear global mode buckling
  // panelStress[0] is the panel in-plane load Nx, panelStress[2] is panel shear
  // in-plane load Nxy NOTE : not using smeared properties here for the local
  // panel stress (closed-form instead for stiffener properties)
  TacsScalar N1GlobalSens, N1CritGlobalSens, N12GlobalSens, N12CritGlobalSens;
  fails[2] = this->bucklingEnvelopeSens(
      -panelStress[0], N1CritGlobal, panelStress[2], N12CritGlobal,
      &N1GlobalSens, &N1CritGlobalSens, &N12GlobalSens, &N12CritGlobalSens);

  // --- #3 - Local panel buckling ---
  TacsScalar N1CritLocal, N12CritLocal;

  // compute the local critical loads
  N1CritLocal =
      computeCriticalLocalAxialLoad(D11p, D22p, rho0Panel, xiPanel, zetaPanel);
  // TODO : add other inputs like rho0 here and ML buckling constraints
  N12CritLocal =
      computeCriticalLocalShearLoad(D11p, D22p, xiPanel, rho0Panel, zetaPanel);

  // panelStress[0] is the panel in-plane load Nx, panelStress[2] is panel shear
  // in-plane load Nxy
  TacsScalar N1LocalSens, N12LocalSens, N1CritLocalSens, N12CritLocalSens;
  fails[3] = this->bucklingEnvelopeSens(
      -panelStress[0], N1CritLocal, panelStress[2], N12CritLocal, &N1LocalSens,
      &N1CritLocalSens, &N12LocalSens, &N12CritLocalSens);

  // --- #4 - Stiffener crippling ---
  // compute the A,B,D matrices of the stiffener
  TacsScalar stiffenerStiffness[NUM_TANGENT_STIFFNESS_ENTRIES],
      stiffenerStress[NUM_STRESSES];
  this->computeStiffenerStiffness(stiffenerStiffness);
  const TacsScalar *As, *Ds;
  this->extractTangentStiffness(stiffenerStiffness, &As, NULL, &Ds, NULL, NULL);
  this->computeStiffenerStress(stiffenerStrain, stiffenerStress);

  // compute stiffener non-dimensional parameters
  TacsScalar D11s = Ds[0], D12s = Ds[1], D22s = Ds[3], D66s = Ds[5];
  TacsScalar A11s = As[0], A66s = As[5];
  TacsScalar rho0Stiff, xiStiff, bStiff, hStiff, genPoiss, zetaStiff;
  bStiff = this->stiffenerHeight;
  hStiff = this->stiffenerThick;
  rho0Stiff = computeAffineAspectRatio(D11s, D22s, a, bStiff);
  xiStiff = computeGeneralizedRigidity(D11s, D22s, D12s, D66s);
  genPoiss = computeGeneralizedPoissonsRatio(D12s, D66s);
  zetaStiff = computeTransverseShearParameter(A66s, A11s, bStiff, hStiff);

  // Compute panel dimensions, material props and non-dimensional parameters
  TacsScalar N1, N1CritCrippling;
  N1CritCrippling = computeStiffenerCripplingLoad(
      D11s, D22s, xiStiff, rho0Stiff, genPoiss, zetaStiff);
  N1 = -stiffenerStress[0];
  fails[4] = N1 / N1CritCrippling;
  // --- End of computeFailuresValues subsections ---

  // Compute the sensitivity of the aggregate failure value to the individual
  // failure mode values
  TacsScalar fail =
      ksAggregationSens(fails, this->NUM_FAILURES, this->ksWeight, dKSdf);

  // Compute the total sensitivity due  to the panel and stiffener material
  // failure
  memset(sens, 0, this->NUM_STRESSES * sizeof(TacsScalar));
  for (int ii = 0; ii < this->NUM_STRESSES; ii++) {
    sens[ii] = dKSdf[0] * panelFailSens[ii] + dKSdf[1] * stiffenerFailSens[ii];
  }

  // Add sensitivites from the buckling criterion
  // local buckling
  N1LocalSens *= dKSdf[2];
  N12LocalSens *= dKSdf[2];
  sens[0] += N1LocalSens * -Ap[0] + N12LocalSens * Ap[2];
  sens[1] += N1LocalSens * -Ap[1] + N12LocalSens * Ap[4];
  sens[2] += N1LocalSens * -Ap[2] + N12LocalSens * Ap[5];

  // Global buckling
  // note here we don't use smeared stiffener properties as closed-form is not
  // based on smearing approach for global buckling now..
  N1GlobalSens *= dKSdf[3];
  N12GlobalSens *= dKSdf[3];
  sens[0] += N1GlobalSens * -Ap[0] + N12GlobalSens * Ap[2];
  sens[1] += N1GlobalSens * -Ap[1] + N12GlobalSens * Ap[4];
  sens[2] += N1GlobalSens * -Ap[2] + N12GlobalSens * Ap[5];

  // Stiffener crippling section:
  TacsScalar N1stiffSens = dKSdf[4] / N1CritCrippling;
  sens[0] += N1stiffSens * -As[0];
  sens[1] += N1stiffSens * -As[1];
  sens[2] += N1stiffSens * -As[2];

  return fail;
}

// Add the derivative of the failure criteria w.r.t. the design variables
void TACSGPBladeStiffenedShellConstitutive::addFailureDVSens(
    int elemIndex, TacsScalar scale, const double pt[], const TacsScalar X[],
    const TacsScalar strain[], int dvLen, TacsScalar dfdx[]) {
  // TODO : need to complete this function

  const int numDV = this->numDesignVars;

  // Compute the failure values and then compute the
  // sensitivity of the aggregate failure value w.r.t. them
  TacsScalar fails[this->NUM_FAILURES], dKSdf[this->NUM_FAILURES];
  TacsScalar fail = this->computeFailureValues(strain, fails);
  ksAggregationSens(fails, this->NUM_FAILURES, this->ksWeight, dKSdf);

  // Sensitivity of the panel failure value to it's DVs
  this->addPanelFailureDVSens(strain, scale * dKSdf[0],
                              &dfdx[this->panelDVStartNum]);

  // Add the direct sensitivity of the stiffener failure value w.r.t DVs
  // Sensitivity of the panel failure value to it's DVs
  TacsScalar stiffenerStrain[TACSBeamConstitutive::NUM_STRESSES];
  this->transformStrain(strain, stiffenerStrain);
  this->addStiffenerFailureDVSens(stiffenerStrain, scale * dKSdf[1],
                                  &dfdx[this->stiffenerDVStartNum]);

  // Add the sensitivity of the stiffener failure value w.r.t. the DVs
  // due to the dependence of the stiffener strains on the DVs
  TacsScalar stiffenerFailStrainSens[TACSBeamConstitutive::NUM_STRESSES];
  this->evalStiffenerFailureStrainSens(stiffenerStrain,
                                       stiffenerFailStrainSens);
  this->addStrainTransformProductDVsens(stiffenerFailStrainSens, strain,
                                        scale * dKSdf[1], dfdx);

  // -- prelim to buckling constraints --
  // compute the A,B,D matrices of the panel
  TacsScalar panelStiffness[NUM_TANGENT_STIFFNESS_ENTRIES],
      panelStress[NUM_STRESSES];
  this->computePanelStiffness(panelStiffness);
  const TacsScalar *Ap, *Dp;
  this->extractTangentStiffness(panelStiffness, &Ap, NULL, &Dp, NULL, NULL);
  this->computePanelStress(strain, panelStress);

  // Compute panel dimensions, material props and non-dimensional parameters
  TacsScalar N1CritGlobal, N12CritGlobal;
  TacsScalar D11p = Dp[0], D12p = Dp[1], D22p = Dp[3], D66p = Dp[5];
  TacsScalar A11p = Ap[0], A66p = Ap[5];
  TacsScalar delta, rho0Panel, xiPanel, gamma, a, b, zetaPanel;
  a = this->panelLength;
  b = this->panelWidth;
  delta = computeStiffenerAreaRatio();
  rho0Panel = computeAffineAspectRatio(D11p, D22p, a, b);
  xiPanel = computeGeneralizedRigidity(D11p, D22p, D12p, D66p);
  gamma = computeStiffenerStiffnessRatio(D11p);
  zetaPanel = computeTransverseShearParameter(A66p, A11p, b, this->panelThick);

  // --- #2 - Global panel buckling ---
  // ----------------------------------------------------------------------------

  // previous section writes directly into dfdx, this section writes into DVsens
  // first and then into dfdx at the end DV parameter sens [0 - panel length, 1
  // - stiff pitch, 2 - panel thick,
  //                    3 - stiff height, 4 - stiff thick, 5 - panel width]
  TacsScalar* DVsens = new TacsScalar[6];
  memset(DVsens, 0, 6 * sizeof(TacsScalar));

  // set initial A,D matrix, nondim parameter and DV sensitivities to
  // backpropagate to.
  TacsScalar* Dpsens = new TacsScalar[4];  // D11,D12,D22,D66
  TacsScalar* Apsens = new TacsScalar[4];  // A11,A12,A22,A66
  memset(Dpsens, 0, 4 * sizeof(TacsScalar));
  memset(Apsens, 0, 4 * sizeof(TacsScalar));
  // ND parameter sens [xi, rho0, delta, gamma, zeta]
  TacsScalar* NDsens = new TacsScalar[5];
  memset(NDsens, 0, 5 * sizeof(TacsScalar));

  // compute the global critical loads
  N1CritGlobal = computeCriticalGlobalAxialLoad(D11p, D22p, b, delta, rho0Panel,
                                                xiPanel, gamma, zetaPanel);
  N12CritGlobal = computeCriticalGlobalShearLoad(D11p, D22p, b, xiPanel,
                                                 rho0Panel, gamma, zetaPanel);

  // backpropagate from fails[2] -> through the buckling envelope on
  // N11/N11crit, N12/N12crit
  TacsScalar N1GlobalSens, N1CritGlobalSens, N12GlobalSens, N12CritGlobalSens;
  fails[2] = this->bucklingEnvelopeSens(
      -panelStress[0], N1CritGlobal, panelStress[2], N12CritGlobal,
      &N1GlobalSens, &N1CritGlobalSens, &N12GlobalSens, &N12CritGlobalSens);

  // multiply the dKS/dfails[2] jacobian to complete this step
  N1GlobalSens *= dKSdf[2];
  N1CritGlobalSens *= dKSdf[2];
  N12GlobalSens *= dKSdf[2];
  N12CritGlobalSens *= dKSdf[2];

  // backpropagate the in plane load sensitivities to DVs
  // Convert sensitivity w.r.t applied loads into sensitivity w.r.t DVs
  TacsScalar dfdPanelStress[NUM_STRESSES];
  memset(dfdPanelStress, 0, NUM_STRESSES * sizeof(TacsScalar));
  dfdPanelStress[0] = -N1GlobalSens;
  dfdPanelStress[2] = N12GlobalSens;
  // figure out whether this should be based on
  this->addPanelStressDVSens(scale, strain, dfdPanelStress,
                             &dfdx[this->panelDVStartNum]);

  // backpropagate crit load sensitivities through the critical load computation
  computeCriticalGlobalAxialLoadSens(
      N1CritGlobalSens, D11p, D22p, b, delta, rho0Panel, xiPanel, gamma,
      zetaPanel, &Dpsens[0], &Dpsens[2], &DVsens[5], &NDsens[2], &NDsens[1],
      &NDsens[0], &NDsens[3], &NDsens[4]);
  computeCriticalGlobalShearLoadSens(N12CritGlobalSens, D11p, D22p, b, xiPanel,
                                     rho0Panel, gamma, zetaPanel, &Dpsens[0],
                                     &Dpsens[2], &DVsens[5], &NDsens[0],
                                     &NDsens[1], &NDsens[3], &NDsens[4]);

  // --- #3 - Local Panel Buckling ---
  // ----------------------------------------------------------------------------

  TacsScalar N1CritLocal, N12CritLocal;

  // compute the local critical loads
  N1CritLocal =
      computeCriticalLocalAxialLoad(D11p, D22p, rho0Panel, xiPanel, zetaPanel);
  N12CritLocal =
      computeCriticalLocalShearLoad(D11p, D22p, xiPanel, rho0Panel, zetaPanel);

  // backpropagate from fails[3] -> through the buckling envelope on
  // N11/N11crit, N12/N12crit
  TacsScalar N1LocalSens, N1CritLocalSens, N12LocalSens, N12CritLocalSens;
  fails[3] = this->bucklingEnvelopeSens(
      -panelStress[0], N1CritLocal, panelStress[2], N12CritLocal, &N1LocalSens,
      &N1CritLocalSens, &N12LocalSens, &N12CritLocalSens);

  // multiply the dKS/dfails[3] jacobian to complete this step
  N1LocalSens *= dKSdf[3];
  N1CritLocalSens *= dKSdf[3];
  N12LocalSens *= dKSdf[3];
  N12CritLocalSens *= dKSdf[3];

  // backpropagate the in plane load sensitivities to DVs
  // Convert sensitivity w.r.t applied loads into sensitivity w.r.t DVs
  memset(dfdPanelStress, 0, NUM_STRESSES * sizeof(TacsScalar));
  dfdPanelStress[0] = -N1LocalSens;
  dfdPanelStress[2] = N12LocalSens;
  // figure out whether this should be based on
  this->addPanelStressDVSens(scale, strain, dfdPanelStress,
                             &dfdx[this->panelDVStartNum]);

  // backpropagate crit load sensitivities through the critical load computation
  computeCriticalLocalAxialLoadSens(
      N1CritLocalSens, D11p, D22p, rho0Panel, xiPanel, zetaPanel, &Dpsens[0],
      &Dpsens[2], &NDsens[1], &NDsens[0], &DVsens[1], &NDsens[4]);
  computeCriticalLocalShearLoadSens(
      N12CritLocalSens, D11p, D22p, xiPanel, rho0Panel, zetaPanel, &Dpsens[0],
      &Dpsens[2], &DVsens[1], &NDsens[0], &NDsens[1], &NDsens[4]);

  // 2 & 3 - backpropagate ND parameters of the panel to the DVs
  // -----------------------------------------------------------
  // NDsens for panel, [xi, rho0, delta, gamma, zeta]

  computeGeneralizedRigiditySens(NDsens[0], D11p, D22p, D12p, D66p, &Dpsens[0],
                                 &Dpsens[2], &Dpsens[1], &Dpsens[3]);
  computeAffineAspectRatioSens(NDsens[1], D11p, D22p, a, b, &Dpsens[0],
                               &Dpsens[2], &DVsens[0], &DVsens[5]);
  computeStiffenerAreaRatioSens(NDsens[2], &DVsens[4], &DVsens[3], &DVsens[1],
                                &DVsens[2]);
  computeStiffenerStiffnessRatioSens(NDsens[3], D11p, &Dpsens[0], &DVsens[4],
                                     &DVsens[3], &DVsens[1]);
  computeTransverseShearParameterSens(NDsens[4], A66p, A11p, b,
                                      this->panelThick, &Apsens[3], &Apsens[0],
                                      &DVsens[5], &DVsens[2]);

  // 2 & 3 - backpropagate A & D matrix sensitivities back to the DVs
  // --------------------------------------------------------------
  if (this->panelThickNum >= 0) {
    int dvNum = this->panelThickLocalNum;
    TacsScalar t = this->panelThick;

    // backpropagate through the D matrix
    TacsScalar dMatdt = 0.25 * t * t;  // d/dt(t^3/12) = t^2/4
    for (int ii = 0; ii < this->numPanelPlies; ii++) {
      TacsScalar* Q = &(this->panelQMats[ii * NUM_Q_ENTRIES]);
      dfdx[dvNum] += scale * dMatdt * this->panelPlyFracs[ii] *
                     (Dpsens[0] * Q[0] + Dpsens[2] * Q[3] + Dpsens[1] * Q[1] +
                      Dpsens[3] * Q[5]);
    }

    // backpropagate through the A matrix
    for (int ii = 0; ii < this->numPanelPlies; ii++) {
      TacsScalar* Q = &(this->panelQMats[ii * NUM_Q_ENTRIES]);
      dfdx[dvNum] += scale * this->panelPlyFracs[ii] *
                     (Apsens[0] * Q[0] + Apsens[2] * Q[3] + Apsens[1] * Q[1] +
                      Apsens[3] * Q[5]);
    }
  }

  // --- Panel Ply fraction sensitivities ---
  // TBD

  // --- #4 - Stiffener crippling ---
  // ----------------------------------------------------

  // setup new ND parameter array for stiffener computation
  // ND parameter sens [xi, rho0, genPoiss, zeta]
  TacsScalar* stiffNDsens = new TacsScalar[4];
  memset(stiffNDsens, 0, 4 * sizeof(TacsScalar));

  // set initial A,D matrix, nondim parameter and DV sensitivities to
  // backpropagate to.
  TacsScalar* Dssens = new TacsScalar[4];  // D11,D12,D22,D66
  TacsScalar* Assens = new TacsScalar[4];  // A11,A12,A22,A66
  memset(Dssens, 0, 4 * sizeof(TacsScalar));
  memset(Assens, 0, 4 * sizeof(TacsScalar));

  // compute the A,B,D matrices of the stiffener
  TacsScalar stiffenerStiffness[NUM_TANGENT_STIFFNESS_ENTRIES],
      stiffenerStress[NUM_STRESSES];
  this->computeStiffenerStiffness(stiffenerStiffness);
  const TacsScalar *As, *Ds;
  this->extractTangentStiffness(stiffenerStiffness, &As, NULL, &Ds, NULL, NULL);
  this->computeStiffenerStress(stiffenerStrain, stiffenerStress);

  // compute stiffener non-dimensional parameters
  TacsScalar D11s = Ds[0], D12s = Ds[1], D22s = Ds[3], D66s = Ds[5];
  TacsScalar A11s = As[0], A66s = As[5];
  TacsScalar rho0Stiff, xiStiff, bStiff, hStiff, genPoiss, zetaStiff;
  bStiff = this->stiffenerHeight;
  hStiff = this->stiffenerThick;
  rho0Stiff = computeAffineAspectRatio(D11s, D22s, a, bStiff);
  xiStiff = computeGeneralizedRigidity(D11s, D22s, D12s, D66s);
  genPoiss = computeGeneralizedPoissonsRatio(D12s, D66s);
  zetaStiff = computeTransverseShearParameter(A66s, A11s, bStiff, hStiff);

  // Compute panel dimensions, material props and non-dimensional parameters
  TacsScalar N1, N1CritCrippling;
  N1CritCrippling = computeStiffenerCripplingLoad(
      D11s, D22s, xiStiff, rho0Stiff, genPoiss, zetaStiff);
  N1 = -stiffenerStress[0];
  fails[4] = N1 / N1CritCrippling;
  // --- End of computeFailuresValues subsections ---

  // backpropagate from fails[4] to N1 stiffener in plane load
  TacsScalar stiffN1sens = dKSdf[4] * fails[4] / N1;
  TacsScalar stiffN1critsens = dKSdf[4] * fails[4] * -1.0 / N1CritCrippling;

  // backpropagate from N1 in plane load to material sensitivities for the
  // stiffener
  TacsScalar dfdStiffStress[NUM_STRESSES];
  memset(dfdStiffStress, 0, NUM_STRESSES * sizeof(TacsScalar));
  dfdStiffStress[0] = -stiffN1sens;
  this->addStiffenerStressDVSens(scale, strain, dfdStiffStress,
                                 &dfdx[this->panelDVStartNum]);

  // backpropagate N1crit sens of stiffener to ND and material sensitivities
  computeStiffenerCripplingLoadSens(
      stiffN1critsens, D11s, D22s, xiStiff, rho0Stiff, genPoiss, zetaStiff,
      &Dssens[0], &Dssens[2], &DVsens[3], &stiffNDsens[0], &stiffNDsens[1],
      &stiffNDsens[2], &stiffNDsens[3]);

  // backpropagate ND sensitivities to A, D matrix for stiffener and DV sens
  computeGeneralizedRigiditySens(stiffNDsens[0], D11s, D22s, D12s, D66s,
                                 &Dssens[0], &Dssens[2], &Dssens[1],
                                 &Dssens[3]);
  computeAffineAspectRatioSens(stiffNDsens[1], D11s, D22s, a, bStiff,
                               &Dssens[0], &Dssens[2], &DVsens[0], &DVsens[3]);
  computeGeneralizedPoissonsRatioSens(stiffNDsens[2], D12s, D66s, &Dssens[1],
                                      &Dssens[3]);
  computeTransverseShearParameterSens(stiffNDsens[3], A66s, A11s, bStiff,
                                      hStiff, &Assens[3], &Assens[0],
                                      &DVsens[3], &DVsens[4]);

  // backpropagate stiffener A,D matrix sensitivities through stiffener DV
  // -----------------------

  if (this->stiffenerThickLocalNum >= 0) {
    int dvNum = this->stiffenerThickLocalNum;
    TacsScalar t = this->stiffenerThick;

    // backpropagate through the D matrix
    TacsScalar dMatdt = 0.25 * t * t;  // d/dt(t^3/12) = t^2/4
    for (int ii = 0; ii < this->numStiffenerPlies; ii++) {
      TacsScalar* Q = &(this->stiffenerQMats[ii * NUM_Q_ENTRIES]);
      dfdx[dvNum] += scale * dMatdt * this->stiffenerPlyFracs[ii] *
                     (Dssens[0] * Q[0] + Dssens[2] * Q[3] + Dssens[1] * Q[1] +
                      Dssens[3] * Q[5]);
    }

    // backpropagate through the A matrix
    for (int ii = 0; ii < this->numStiffenerPlies; ii++) {
      TacsScalar* Q = &(this->stiffenerQMats[ii * NUM_Q_ENTRIES]);
      dfdx[dvNum] += scale * this->stiffenerPlyFracs[ii] *
                     (Assens[0] * Q[0] + Assens[2] * Q[3] + Assens[1] * Q[1] +
                      Assens[3] * Q[5]);
    }
  }

  // TBD stiffener ply fraction variables
  // Just haven't written this part yet.. since not using it yet
}

// Retrieve the design variable for plotting purposes
TacsScalar TACSGPBladeStiffenedShellConstitutive::evalDesignFieldValue(
    int elemIndex, const double pt[], const TacsScalar X[], int index) {
  switch (index) {
    case 0:
      return this->computeEffectiveThickness();
    case 1:
      return this->computeEffectiveBendingThickness();
    case 2:
      return this->panelLength;
    case 3:
      return this->stiffenerPitch;
    case 4:
      return this->panelThick;
    case 5:
      return this->stiffenerHeight;
    case 6:
      return this->stiffenerThick;
    case 7:
      return this->panelWidth;
  }
  return 0.0;
}

// Retrieve the global design variable numbers
int TACSGPBladeStiffenedShellConstitutive::getDesignVarNums(int elemIndex,
                                                            int dvLen,
                                                            int dvNums[]) {
  TACSBladeStiffenedShellConstitutive::getDesignVarNums(elemIndex, dvLen,
                                                        dvNums);
  if (dvNums && dvLen >= this->numDesignVars) {
    if (this->panelWidthNum >= 0) {
      dvNums[this->panelWidthLocalNum] = panelWidthNum;
    }
  }
  return numDesignVars;
}

// Set the element design variable from the design vector
int TACSGPBladeStiffenedShellConstitutive::setDesignVars(
    int elemIndex, int dvLen, const TacsScalar dvs[]) {
  TACSBladeStiffenedShellConstitutive::setDesignVars(elemIndex, dvLen, dvs);
  if (dvLen >= this->numDesignVars) {
    if (this->panelWidthNum >= 0) {
      this->panelWidth = dvs[this->panelWidthLocalNum];
    }
  }
  return this->numDesignVars;
}

// Get the element design variables values
int TACSGPBladeStiffenedShellConstitutive::getDesignVars(int elemIndex,
                                                         int dvLen,
                                                         TacsScalar dvs[]) {
  TACSBladeStiffenedShellConstitutive::getDesignVars(elemIndex, dvLen, dvs);
  if (dvLen >= this->numDesignVars) {
    if (this->panelWidthNum >= 0) {
      dvs[this->panelWidthLocalNum] = this->panelWidth;
    }
  }
  return this->numDesignVars;
}

// Get the lower and upper bounds for the design variable values
int TACSGPBladeStiffenedShellConstitutive::getDesignVarRange(int elemIndex,
                                                             int dvLen,
                                                             TacsScalar lb[],
                                                             TacsScalar ub[]) {
  TACSBladeStiffenedShellConstitutive::getDesignVarRange(elemIndex, dvLen, lb,
                                                         ub);
  if (dvLen >= this->numDesignVars) {
    if (this->panelWidthNum >= 0) {
      lb[this->panelWidthLocalNum] = this->panelWidthLowerBound;
      ub[this->panelWidthLocalNum] = this->panelWidthUpperBound;
    }
  }
  return this->numDesignVars;
}

// ==============================================================================
// Buckling functions
// ==============================================================================

TacsScalar TACSGPBladeStiffenedShellConstitutive::computeAffineAspectRatioSens(
    const TacsScalar rho0sens, const TacsScalar D11, const TacsScalar D22,
    const TacsScalar a, const TacsScalar b, TacsScalar* D11sens,
    TacsScalar* D22sens, TacsScalar* asens, TacsScalar* bsens) {
  // compute the derivatives of the affine aspect ratio and return the affine
  // aspect ratio
  TacsScalar rho_0 = computeAffineAspectRatio(D11, D22, a, b);
  // where rho_0 = a/b * (D22/D11)**0.25

  // use power series rules d(x^p) = p * (x^p) / x to cleanly differentiate the
  // expression
  *asens = rho0sens * rho_0 / a;
  *bsens = rho0sens * -1.0 * rho_0 / b;
  *D11sens = rho0sens * -0.25 * rho_0 / D11;
  *D22sens = rho0sens * 0.25 * rho_0 / D22;

  return rho_0;
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeGeneralizedRigiditySens(
    const TacsScalar xisens, const TacsScalar D11, const TacsScalar D22,
    const TacsScalar D12, const TacsScalar D66, TacsScalar* D11sens,
    TacsScalar* D22sens, TacsScalar* D12sens, TacsScalar* D66sens) {
  // compute the derivatives of the generalized rigidity xi
  TacsScalar denominator = sqrt(D11 * D22);
  TacsScalar xi = computeGeneralizedRigidity(D11, D22, D12, D66);
  // so that xi = (D12 + 2 * D66) / sqrt(D11*D22)

  // compute the sensitivities
  *D12sens += xisens * 1.0 / denominator;
  *D66sens += xisens * 2.0 / denominator;
  *D11sens += xisens * -0.5 * xi / D11;
  *D22sens += xisens * -0.5 * xi / D22;

  return xi;
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeGeneralizedPoissonsRatioSens(
    const TacsScalar epssens, const TacsScalar D12, const TacsScalar D66,
    TacsScalar* D12sens, TacsScalar* D66sens) {
  // compute derivatives of the generalized poisson's ratio
  TacsScalar eps = computeGeneralizedPoissonsRatio(D12, D66);
  // where eps = (D12 + 2 * D66) / D12

  *D12sens += epssens * -2.0 * D66 / D12 / D12;
  *D66sens += epssens * 2.0 / D12;

  return eps;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::computeStiffenerAreaRatio() {
  // get effective moduli for the panel and stiffener
  TacsScalar E1s, E1p, _;
  // first the effective modulus of the panel/plate
  // need to add derivatives w.r.t. panel plies here then..
  this->computeEffectiveModulii(this->numPanelPlies, this->panelQMats,
                                this->panelPlyFracs, &E1p, &_);

  // then the stiffener
  this->computeEffectiveModulii(this->numStiffenerPlies, this->stiffenerQMats,
                                this->stiffenerPlyFracs, &E1s, &_);

  TacsScalar As = this->computeStiffenerArea();
  return E1s * As / (E1p * this->stiffenerPitch * this->panelThick);
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::computeStiffenerAreaRatioSens(
    const TacsScalar deltasens, TacsScalar* sthickSens, TacsScalar* sheightSens,
    TacsScalar* spitchSens, TacsScalar* pthickSens) {
  // get effective moduli for the panel and stiffener
  TacsScalar delta = this->computeStiffenerAreaRatio();

  // TODO : may need to add the derivatives w.r.t. panel ply fractions for E1p
  // computation later..
  *sthickSens += deltasens * delta / this->stiffenerThick;
  *sheightSens += deltasens * delta / this->stiffenerHeight;
  *spitchSens += deltasens * -1.0 * delta / this->stiffenerPitch;
  *pthickSens += deltasens * -1.0 * delta / this->panelThick;

  return delta;
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeStiffenerStiffnessRatio(
    TacsScalar D11) {
  // get effective moduli for the panel and stiffener
  TacsScalar E1s, E1p, _;
  // first the effective modulus of the panel/plate
  // need to add derivatives w.r.t. panel plies here then..
  this->computeEffectiveModulii(this->numPanelPlies, this->panelQMats,
                                this->panelPlyFracs, &E1p, &_);

  // then the stiffener
  this->computeEffectiveModulii(this->numStiffenerPlies, this->stiffenerQMats,
                                this->stiffenerPlyFracs, &E1s, &_);

  // get the stiffener bending inertia in 11-direction
  // TODO : double check if this bending stiffness calculation is correct..
  TacsScalar Is = this->computeStiffenerIzz();

  return E1s * Is / D11 / this->stiffenerThick;
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeStiffenerStiffnessRatioSens(
    const TacsScalar gammasens, const TacsScalar D11, TacsScalar* D11sens,
    TacsScalar* sthickSens, TacsScalar* sheightSens, TacsScalar* spitchSens) {
  // use power series rules and the forward state to differentiate
  TacsScalar gamma = computeStiffenerStiffnessRatio(
      D11);  // this is the stiffener stiffness ratio (final forward state)

  // intermediate states and sensitivities
  TacsScalar Is = this->computeStiffenerIzz();
  TacsScalar dI_dsthick, dI_dsheight;
  this->computeStiffenerIzzSens(dI_dsthick, dI_dsheight);

  // TODO : may need to add stiffener ply fraction derivatives for E1p
  // computation, for now ignoring
  *D11sens += gammasens * -1.0 * gamma / D11;
  *sthickSens += gammasens * gamma / Is * dI_dsthick;
  *sheightSens += gammasens * gamma / Is * dI_dsheight;
  *spitchSens += gammasens * -1.0 * gamma / this->stiffenerPitch;

  return gamma;
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeTransverseShearParameter(
    TacsScalar A66, TacsScalar A11, TacsScalar b, TacsScalar h) {
  return A66 / A11 * (b / h) * (b / h);
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeTransverseShearParameterSens(
    const TacsScalar zetasens, const TacsScalar A66, const TacsScalar A11,
    const TacsScalar b, const TacsScalar h, TacsScalar* A66sens,
    TacsScalar* A11sens, TacsScalar* bsens, TacsScalar* hsens) {
  TacsScalar zeta = A66 / A11 * (b / h) * (b / h);
  TacsScalar dzeta = zetasens * zeta;

  *A66sens += dzeta / A66;
  *A11sens += dzeta * -1.0 / A11;
  *bsens += dzeta * 2.0 / b;
  *hsens += dzeta * -2.0 / h;

  return zeta;
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeCriticalGlobalAxialLoad(
    const TacsScalar D11, const TacsScalar D22, const TacsScalar b,
    const TacsScalar delta, const TacsScalar rho_0, const TacsScalar xi,
    const TacsScalar gamma, const TacsScalar zeta) {
  if (this->getAxialGP()) {
    // use Gaussian processes to compute the critical global axial load
    TacsScalar dim_factor =
        M_PI * M_PI * sqrt(D11 * D22) / b / b / (1.0 + delta);
    TacsScalar* Xtest = new TacsScalar[this->getAxialGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = log(1 + gamma);
    Xtest[3] = log(zeta);
    return dim_factor * exp(this->getAxialGP()->predictMeanTestData(Xtest));

  } else {
    // use the CPT closed-form solution to compute the critical global axial
    // load
    TacsScalar neg_N11crits[this->NUM_CF_MODES];
    for (int m1 = 1; m1 < this->NUM_CF_MODES + 1; m1++) {
      TacsScalar dim_factor =
          M_PI * M_PI * sqrt(D11 * D22) / b / b / (1.0 + delta);
      TacsScalar nondim_factor = (1.0 + gamma) * pow(m1 / rho_0, 2.0) +
                                 pow(m1 / rho_0, -2.0) + 2.0 * xi;
      neg_N11crits[m1 - 1] =
          -1.0 * dim_factor * nondim_factor;  // negated only because we have to
                                              // do KS min aggregate later
    }

    // compute KS aggregation for -N11crit for each mode then negate again
    // (because we want minimum N11crit so maximize negative N11crit)
    TacsScalar neg_N11crit =
        ksAggregation(neg_N11crits, this->NUM_CF_MODES, this->ksWeight);
    return -1.0 * neg_N11crit;
  }
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeCriticalGlobalAxialLoadSens(
    const TacsScalar N1sens, const TacsScalar D11, const TacsScalar D22,
    const TacsScalar b, const TacsScalar delta, const TacsScalar rho_0,
    const TacsScalar xi, const TacsScalar gamma, const TacsScalar zeta,
    TacsScalar* D11sens, TacsScalar* D22sens, TacsScalar* bsens,
    TacsScalar* deltasens, TacsScalar* rho_0sens, TacsScalar* xisens,
    TacsScalar* gammasens, TacsScalar* zetasens) {
  if (this->getAxialGP()) {
    // use Gaussian processes to compute the critical global axial load
    TacsScalar dim_factor =
        M_PI * M_PI * sqrt(D11 * D22) / b / b / (1.0 + delta);
    TacsScalar* Xtest = new TacsScalar[this->getAxialGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = log(1 + gamma);
    Xtest[3] = log(zeta);
    TacsScalar arg = this->getAxialGP()->predictMeanTestData(Xtest);
    TacsScalar nondim_factor = exp(arg);
    TacsScalar output = dim_factor * nondim_factor;

    // compute sensitivities backwards propagated out of the GP
    // back to the nondim parameter inputs, (this part differentiates the
    // nondim_factor)
    int n_axial_param = this->getAxialGP()->getNparam();
    TacsScalar* Xtestsens = new TacsScalar[n_axial_param];
    TacsScalar Ysens = N1sens * output;
    this->getAxialGP()->predictMeanTestDataSens(Ysens, Xtest, Xtestsens);
    *xisens += Xtestsens[0] / xi;  // chain rule dlog(xi)/dxi = 1/xi
    *rho_0sens += Xtestsens[1] / rho_0;
    *gammasens += Xtestsens[2] / (1.0 + gamma);
    *zetasens += Xtestsens[3] / zeta;

    // compute the sensivities of inputs in the dimensional constant
    // (this part differentiates the dim factor)
    *D11sens += Ysens * 0.5 / D11;
    *D22sens += Ysens * 0.5 / D22;
    *bsens += Ysens * -2.0 / b;
    *deltasens += Ysens * -1.0 / (1.0 + delta);
    return output;

  } else {
    // use the CPT closed-form solution to compute the critical global axial
    // load forward analysis part here
    TacsScalar neg_N11crits[this->NUM_CF_MODES];
    for (int m1 = 1; m1 < this->NUM_CF_MODES + 1; m1++) {
      TacsScalar dim_factor =
          M_PI * M_PI * sqrt(D11 * D22) / b / b / (1.0 + delta);
      TacsScalar nondim_factor = (1.0 + gamma) * pow(m1 / rho_0, 2.0) +
                                 pow(m1 / rho_0, -2.0) + 2.0 * xi;
      neg_N11crits[m1 - 1] =
          -1.0 * dim_factor * nondim_factor;  // negated only because we have to
                                              // do KS min aggregate later
    }

    // compute KS aggregation sensitivity
    TacsScalar neg_N11crits_sens[this->NUM_CF_MODES];
    TacsScalar neg_N11crit = ksAggregationSens(
        neg_N11crits, this->NUM_CF_MODES, this->ksWeight, neg_N11crits_sens);

    // compute sensitivities here
    for (int m1 = 1; m1 < this->NUM_CF_MODES + 1; m1++) {
      // apply output sens
      neg_N11crits_sens[m1 - 1] *= N1sens;

      // forward analysis states
      TacsScalar dim_factor =
          M_PI * M_PI * sqrt(D11 * D22) / b / b / (1.0 + delta);
      TacsScalar nondim_factor = (1.0 + gamma) * pow(m1 / rho_0, 2.0) +
                                 pow(m1 / rho_0, -2.0) + 2.0 * xi;
      neg_N11crits[m1 - 1] =
          -1.0 * dim_factor * nondim_factor;  // negated only because we have to
                                              // do KS min aggregate later

      // update sensitivities (left factor is dKS/dv_i, right factor is dv_i /
      // dx)
      *D11sens +=
          neg_N11crits_sens[m1 - 1] * (0.5 * neg_N11crits[m1 - 1] / D11);
      *D22sens +=
          neg_N11crits_sens[m1 - 1] * (0.5 * neg_N11crits[m1 - 1] / D22);
      *bsens += neg_N11crits_sens[m1 - 1] * (-2.0 * neg_N11crits[m1 - 1] / b);
      *deltasens += neg_N11crits_sens[m1 - 1] *
                    (-1.0 * neg_N11crits[m1 - 1] / (1.0 + delta));
      *rho_0sens += neg_N11crits_sens[m1 - 1] * -dim_factor *
                    ((1.0 + gamma) * -2.0 * pow(m1 / rho_0, 2.0) / rho_0 +
                     pow(m1 / rho_0, -2.0) * 2.0 / rho_0);
      *xisens += neg_N11crits_sens[m1 - 1] * -dim_factor * 2.0;
      *gammasens +=
          neg_N11crits_sens[m1 - 1] * -dim_factor * pow(m1 / rho_0, 2.0);
    }
    return -1.0 * neg_N11crit;
  }
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::computeCriticalLocalAxialLoad(
    const TacsScalar D11, const TacsScalar D22, const TacsScalar rho_0,
    const TacsScalar xi, const TacsScalar zeta) {
  if (this->getAxialGP()) {
    // use Gaussian processes to compute the critical global axial load
    TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                            this->stiffenerPitch / this->stiffenerPitch;
    TacsScalar* Xtest = new TacsScalar[this->getAxialGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = 0.0;  // log(1+gamma) = 0 since gamma=0 for unstiffened panel
    Xtest[3] = log(zeta);
    return dim_factor * exp(this->getAxialGP()->predictMeanTestData(Xtest));

  } else {
    // use the CPT closed-form solution to compute the critical global axial
    // load
    TacsScalar neg_N11crits[this->NUM_CF_MODES];
    for (int m1 = 1; m1 < this->NUM_CF_MODES + 1; m1++) {
      TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                              this->stiffenerPitch / this->stiffenerPitch;
      TacsScalar nondim_factor =
          pow(m1 / rho_0, 2.0) + pow(m1 / rho_0, -2.0) + 2.0 * xi;
      neg_N11crits[m1 - 1] =
          -1.0 * dim_factor * nondim_factor;  // negated only because we have to
                                              // do KS min aggregate later
    }

    // compute KS aggregation for -N11crit for each mode then negate again
    // (because we want minimum N11crit so maximize negative N11crit)
    TacsScalar neg_N11crit =
        ksAggregation(neg_N11crits, this->NUM_CF_MODES, this->ksWeight);
    return -1.0 * neg_N11crit;
  }
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeCriticalLocalAxialLoadSens(
    const TacsScalar N1sens, const TacsScalar D11, const TacsScalar D22,
    const TacsScalar rho_0, const TacsScalar xi, const TacsScalar zeta,
    TacsScalar* D11sens, TacsScalar* D22sens, TacsScalar* rho_0sens,
    TacsScalar* xisens, TacsScalar* spitchsens, TacsScalar* zetasens) {
  if (this->getAxialGP()) {
    // use Gaussian processes to compute the critical global axial load
    TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                            this->stiffenerPitch / this->stiffenerPitch;
    TacsScalar* Xtest = new TacsScalar[this->getAxialGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = 0.0;  // log(1+gamma) = 0 since gamma=0 for unstiffened panel
    Xtest[3] = log(zeta);

    TacsScalar arg = this->getAxialGP()->predictMeanTestData(Xtest);
    TacsScalar nondim_factor = exp(arg);
    TacsScalar output = dim_factor * nondim_factor;

    // backwards propagate sensitivities out of the axialGP model to nondim
    // params, (this part differentiates the nondim_factor)
    int n_axial_param = this->getAxialGP()->getNparam();
    TacsScalar* Xtestsens = new TacsScalar[n_axial_param];
    TacsScalar Ysens = N1sens * output;
    this->getAxialGP()->predictMeanTestDataSens(Ysens, Xtest, Xtestsens);

    *xisens += Xtestsens[0] / xi;
    *rho_0sens +=
        Xtestsens[1] / rho_0;  // chain rule dlog(rho_0) / drho_0 = 1/rho_0
    *zetasens += Xtestsens[3] / zeta;

    // backpropagate the dimensional factor terms out to the material and
    // geometric DVs (this part differentiates the dim_factor)
    *D11sens += Ysens * 0.5 / D11;
    *D22sens += Ysens * 0.5 / D22;
    *spitchsens += Ysens * -2.0 / this->stiffenerPitch;

    // return the final forward analysis output
    return output;

  } else {
    // use the CPT closed-form solution to compute the critical global axial
    // load forward analysis part here
    TacsScalar neg_N11crits[this->NUM_CF_MODES];
    for (int m1 = 1; m1 < this->NUM_CF_MODES + 1; m1++) {
      TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                              this->stiffenerPitch / this->stiffenerPitch;
      TacsScalar nondim_factor =
          pow(m1 / rho_0, 2.0) + pow(m1 / rho_0, -2.0) + 2.0 * xi;
      neg_N11crits[m1 - 1] =
          -1.0 * dim_factor * nondim_factor;  // negated only because we have to
                                              // do KS min aggregate later
    }

    // compute KS aggregation sensitivity
    TacsScalar neg_N11crits_sens[this->NUM_CF_MODES];
    TacsScalar neg_N11crit = ksAggregationSens(
        neg_N11crits, this->NUM_CF_MODES, this->ksWeight, neg_N11crits_sens);

    // compute sensitivities here
    *D11sens = *D22sens = *spitchsens = *rho_0sens = *xisens = 0.0;
    for (int m1 = 1; m1 < this->NUM_CF_MODES + 1; m1++) {
      // backpropagate through the output
      neg_N11crits_sens[m1 - 1] *= N1sens;

      // forward analysis states
      TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                              this->stiffenerPitch / this->stiffenerPitch;
      TacsScalar nondim_factor =
          pow(m1 / rho_0, 2.0) + pow(m1 / rho_0, -2.0) + 2.0 * xi;
      neg_N11crits[m1 - 1] =
          -1.0 * dim_factor * nondim_factor;  // negated only because we have to
                                              // do KS min aggregate later

      // update sensitivities (left factor is dKS/dv_i, right factor is dv_i /
      // dx)
      *D11sens +=
          neg_N11crits_sens[m1 - 1] * (0.5 * neg_N11crits[m1 - 1] / D11);
      *D22sens +=
          neg_N11crits_sens[m1 - 1] * (0.5 * neg_N11crits[m1 - 1] / D22);
      *rho_0sens += neg_N11crits_sens[m1 - 1] * -dim_factor *
                    (-2.0 * pow(m1 / rho_0, 2.0) / rho_0 +
                     pow(m1 / rho_0, -2.0) * 2.0 / rho_0);
      *xisens += neg_N11crits_sens[m1 - 1] * -dim_factor * 2.0;
      *spitchsens += neg_N11crits_sens[m1 - 1] *
                     (-2.0 * neg_N11crits[m1 - 1] / this->stiffenerPitch);
    }
    return -1.0 * neg_N11crit;
  }
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeCriticalGlobalShearLoad(
    const TacsScalar D11, const TacsScalar D22, const TacsScalar b,
    const TacsScalar xi, const TacsScalar rho_0, const TacsScalar gamma,
    const TacsScalar zeta) {
  if (this->getShearGP()) {
    // use Gaussian processes to compute the critical global shear load
    TacsScalar dim_factor =
        M_PI * M_PI * pow(D11 * D22 * D22 * D22, 0.25) / b / b;
    TacsScalar* Xtest = new TacsScalar[this->getShearGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = log(
        1.0 + gamma);  // log(1+gamma) = 0 since gamma=0 for unstiffened panel
    Xtest[3] = log(zeta);
    return dim_factor * exp(this->getShearGP()->predictMeanTestData(Xtest));

  } else {
    // use the CPT closed-form solution to compute the critical global axial
    // load no mode switching in this solution.. (only accurate for higher
    // aspect ratios => hence the need for machine learning for the actual
    // solution)
    TacsScalar lam1, lam2;  // lam1bar, lam2bar values
    nondimShearParams(xi, gamma, &lam1, &lam2);
    TacsScalar dim_factor =
        M_PI * M_PI * pow(D11 * D22 * D22 * D22, 0.25) / b / b;
    TacsScalar nondim_factor =
        (1.0 + pow(lam1, 4.0) + 6.0 * pow(lam1 * lam2, 2.0) + pow(lam2, 4.0) +
         2.0 * xi) /
        (2.0 * lam1 * lam1 * lam2);
    return dim_factor *
           nondim_factor;  // aka N12_crit from CPT closed-form solution
  }
}

void TACSGPBladeStiffenedShellConstitutive::nondimShearParams(
    const TacsScalar xi, const TacsScalar gamma, TacsScalar* lam1bar,
    TacsScalar* lam2bar) {
  // need to iterate over lam2 with the Newton's method
  TacsScalar lam2bar_sq = 0.0;  // starting guess for lambda2_bar^2

  // Newton iteration for lam2bar squared
  while (abs(TacsRealPart(lam2Constraint(lam2bar_sq, xi, gamma))) > 1e-10) {
    lam2bar_sq -= lam2Constraint(lam2bar_sq, xi, gamma) /
                  lam2ConstraintDeriv(lam2bar_sq, xi, gamma);
  }

  // now compute lam1_bar, lam2_bar
  *lam1bar =
      pow(1.0 + 2.0 * lam2bar_sq * xi + lam2bar_sq * lam2bar_sq + gamma, 0.25);
  *lam2bar = pow(lam2bar_sq, 0.5);
}
TacsScalar TACSGPBladeStiffenedShellConstitutive::lam2Constraint(
    const TacsScalar lam2sq, const TacsScalar xi, const TacsScalar gamma) {
  // compute the residual of the combined lam1bar, lam2bar constraint but on
  // lam2bar^2
  TacsScalar lam1bar =
      pow(1.0 + 2.0 * lam2sq * xi + lam2sq * lam2sq + gamma, 0.25);
  TacsScalar lam1sq = lam1bar * lam1bar;
  TacsScalar lam14 = lam1sq * lam1sq;
  return lam2sq + lam1sq + xi / 3.0 -
         sqrt((3.0 + xi) / 9.0 + 4.0 / 3.0 * lam1sq * xi + 4.0 / 3.0 * lam14);
}
TacsScalar TACSGPBladeStiffenedShellConstitutive::lam2ConstraintDeriv(
    const TacsScalar lam2sq, const TacsScalar xi, const TacsScalar gamma) {
  // compute the residual derivatives for the lam2bar constraint above w.r.t.
  // the lam2sq input about a lam2sq input
  TacsScalar dfdlam2sq = 1.0;
  TacsScalar temp = 1.0 + 2.0 * lam2sq * xi + lam2sq * lam2sq + gamma;
  TacsScalar lam1 = pow(temp, 0.25);
  TacsScalar lam1sq = lam1 * lam1;
  TacsScalar lam14 = lam1sq * lam1sq;

  TacsScalar term2 =
      sqrt((3.0 + xi) / 9.0 + 4.0 / 3.0 * lam1sq * xi + 4.0 / 3.0 * lam14);
  TacsScalar dlam1_dlam2sq = lam1 * 0.25 / temp * (2.0 + 2.0 * lam2sq);
  TacsScalar dfdlam1 = 2 * lam1 - 0.5 / term2 * 4.0 / 3.0 *
                                      (2.0 * lam1 * xi + 4.0 * lam1 * lam1sq);
  dfdlam2sq += dfdlam1 * dlam1_dlam2sq;
  return dfdlam2sq;
}

void TACSGPBladeStiffenedShellConstitutive::nondimShearParamsSens(
    const TacsScalar xi, const TacsScalar gamma, TacsScalar* lam1bar,
    TacsScalar* lam2bar, TacsScalar* dl1xi, TacsScalar* dl1gamma,
    TacsScalar* dl2xi, TacsScalar* dl2gamma) {
  // get the lam1, lam2 from Newton's method
  TacsScalar lam1, lam2;
  nondimShearParams(xi, gamma, &lam1, &lam2);

  // also send out the lam1bar, lam2bar again
  *lam1bar = lam1;
  *lam2bar = lam2;

  // differentiate the nonlinear constraints from nondimShearParam subroutine
  // sys eqns [A,B;C,D] * [lam1bar_dot, lam2bar_dot] = [E,F] for each of the two
  // derivatives

  TacsScalar exp1 = 1.0 + 2.0 * lam2 * lam2 * xi + pow(lam2, 4.0) + gamma;
  TacsScalar dexp1lam2 = 4.0 * lam2 * xi + 4.0 * lam2 * lam2 * lam2;
  TacsScalar dexp1xi = 2.0 * lam2 * lam2;
  TacsScalar dexp1gamma = 1.0;
  TacsScalar exp2 =
      (3.0 + xi) / 9.0 + 4.0 / 3.0 * (lam1 * lam1 * xi + pow(lam1, 4.0));
  TacsScalar dexp2lam1 =
      4.0 / 3.0 * (2.0 * lam1 * xi + 4.0 * lam1 * lam1 * lam1);
  TacsScalar dexp2xi = 1.0 / 9.0;
  TacsScalar dexp2gamma = 0.0;

  // first for the xi sensitivities
  TacsScalar A1, B1, C1, D1, E1, F1;
  A1 = 1.0;
  B1 = -0.25 * lam1 / exp1 * dexp1lam2;
  E1 = 0.25 * lam1 / exp1 * dexp1xi;
  C1 = 2.0 * lam1 - 0.5 * pow(exp2, -0.5) * dexp2lam1;
  D1 = 2.0 * lam2;
  F1 = -1.0 / 3.0 + 0.5 * pow(exp2, -0.5) * dexp2xi;
  *dl1xi = (D1 * E1 - B1 * F1) / (A1 * D1 - B1 * C1);
  *dl2xi = (A1 * F1 - C1 * E1) / (A1 * D1 - B1 * C1);

  // then for the gamma sensitivities
  TacsScalar A2, B2, C2, D2, E2, F2;
  A2 = A1;
  B2 = B1;
  E2 = 0.25 * lam1 / exp1 * dexp1gamma;
  C2 = C1;
  D2 = D1;
  F2 = -1.0 / 3.0 + 0.5 * pow(exp2, -0.5) * dexp2gamma;
  *dl1gamma = (D2 * E2 - B2 * F2) / (A2 * D2 - B2 * C2);
  *dl2gamma = (A2 * F2 - C2 * E2) / (A2 * D2 - B2 * C2);
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeCriticalGlobalShearLoadSens(
    const TacsScalar N12sens, const TacsScalar D11, const TacsScalar D22,
    const TacsScalar b, const TacsScalar xi, const TacsScalar rho_0,
    const TacsScalar gamma, const TacsScalar zeta, TacsScalar* D11sens,
    TacsScalar* D22sens, TacsScalar* bsens, TacsScalar* xisens,
    TacsScalar* rho_0sens, TacsScalar* gammasens, TacsScalar* zetasens) {
  if (this->getShearGP()) {
    // use Gaussian processes to compute the critical global shear load
    TacsScalar dim_factor =
        M_PI * M_PI * pow(D11 * D22 * D22 * D22, 0.25) / b / b;
    TacsScalar* Xtest = new TacsScalar[this->getShearGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = log(
        1.0 + gamma);  // log(1+gamma) = 0 since gamma=0 for unstiffened panel
    Xtest[3] = log(zeta);
    TacsScalar arg = this->getShearGP()->predictMeanTestData(Xtest);
    TacsScalar nondim_factor = exp(arg);
    TacsScalar output = dim_factor * nondim_factor;

    // backwards propagate sensitivities out of the axialGP model to nondim
    // params, (this part differentiates the nondim_factor)
    int n_shear_param = this->getShearGP()->getNparam();
    TacsScalar* Xtestsens = new TacsScalar[n_shear_param];
    TacsScalar Ysens = N12sens * output;
    this->getShearGP()->predictMeanTestDataSens(Ysens, Xtest, Xtestsens);

    *xisens += Xtestsens[0] / xi;
    *rho_0sens +=
        Xtestsens[1] / rho_0;  // chain rule dlog(rho_0) / drho_0 = 1/rho_0
    *gammasens += Xtestsens[2] / (1.0 + gamma);
    *zetasens += Xtestsens[3] / zeta;

    // backpropagate the dimensional factor terms out to the material and
    // geometric DVs (this part differentiates the dim_factor)
    *D11sens += Ysens * 0.25 / D11;
    *D22sens += Ysens * 0.75 / D22;
    *bsens += Ysens * -2.0 / b;

    // return the final forward analysis output
    return output;

  } else {
    // use the CPT closed-form solution to compute the critical global axial
    // load no mode switching in this solution.. (only accurate for higher
    // aspect ratios => hence the need for machine learning for the actual
    // solution)
    TacsScalar lam1, lam2;  // lam1bar, lam2bar values
    TacsScalar dl1xi, dl2xi, dl1gamma, dl2gamma;

    // compute the derivatives of the nondimensional constraints
    nondimShearParamsSens(xi, gamma, &lam1, &lam2, &dl1xi, &dl1gamma, &dl2xi,
                          &dl2gamma);

    // compute forward analysis states involved in the N12crit load
    TacsScalar dim_factor =
        M_PI * M_PI * pow(D11 * D22 * D22 * D22, 0.25) / b / b;
    TacsScalar num = 1.0 + pow(lam1, 4.0) + 6.0 * pow(lam1 * lam2, 2.0) +
                     pow(lam2, 4.0) + 2.0 * xi;
    TacsScalar den = 2.0 * lam1 * lam1 * lam2;
    TacsScalar nondim_factor = num / den;
    TacsScalar N12crit = dim_factor * nondim_factor;

    // sensitivities for the non_dim factor

    TacsScalar dNDlam1 =
        (4.0 * pow(lam1, 3.0) + 12.0 * lam1 * lam2 * lam2) / den -
        num * 4.0 * lam1 * lam2 / den / den;
    TacsScalar dNDlam2 =
        (4.0 * pow(lam2, 3.0) + 12.0 * lam2 * lam1 * lam1) / den -
        num * 2.0 * lam1 * lam1 / den / den;

    // compute the overall sensitivities
    *D11sens += N12sens * N12crit * 0.25 / D11;
    *D22sens += N12sens * N12crit * 0.75 / D22;
    *bsens += N12sens * N12crit * -2.0 / b;
    *xisens +=
        N12sens * dim_factor * (dNDlam1 * dl1xi + dNDlam2 * dl2xi + 2.0 / den);
    *gammasens +=
        N12sens * dim_factor * (dNDlam1 * dl1gamma + dNDlam2 * dl2gamma);

    // return N12crit from closed-form solution
    return N12crit;
  }
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::computeCriticalLocalShearLoad(
    const TacsScalar D11, const TacsScalar D22, const TacsScalar xi,
    const TacsScalar rho_0, const TacsScalar zeta) {
  if (this->getShearGP()) {
    // use Gaussian processes to compute the critical global shear load
    TacsScalar s_p = this->stiffenerPitch;
    TacsScalar dim_factor =
        M_PI * M_PI * pow(D11 * D22 * D22 * D22, 0.25) / s_p / s_p;
    TacsScalar* Xtest = new TacsScalar[this->getShearGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = 0.0;  // log(1+gamma) = 0 since gamma=0 for unstiffened panel
    Xtest[3] = log(zeta);
    return dim_factor * exp(this->getShearGP()->predictMeanTestData(Xtest));

  } else {
    // use the CPT closed-form solution to compute the critical global axial
    // load no mode switching in this solution.. (only accurate for higher
    // aspect ratios => hence the need for machine learning for the actual
    // solution)
    TacsScalar lam1, lam2;  // lam1bar, lam2bar values
    nondimShearParams(xi, 0.0, &lam1, &lam2);
    TacsScalar dim_factor = M_PI * M_PI * pow(D11 * D22 * D22 * D22, 0.25) /
                            this->stiffenerPitch / this->stiffenerPitch;
    TacsScalar nondim_factor =
        (1.0 + pow(lam1, 4.0) + 6.0 * pow(lam1 * lam2, 2.0) + pow(lam2, 4.0) +
         2.0 * xi) /
        (2.0 * lam1 * lam1 * lam2);
    return dim_factor *
           nondim_factor;  // aka N12_crit from CPT closed-form solution
  }
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeCriticalLocalShearLoadSens(
    const TacsScalar N12sens, const TacsScalar D11, const TacsScalar D22,
    const TacsScalar xi, const TacsScalar rho_0, const TacsScalar zeta,
    TacsScalar* D11sens, TacsScalar* D22sens, TacsScalar* spitchsens,
    TacsScalar* xisens, TacsScalar* rho_0sens, TacsScalar* zetasens) {
  if (this->getShearGP()) {
    // use Gaussian processes to compute the critical global shear load
    TacsScalar s_p = this->stiffenerPitch;
    TacsScalar dim_factor =
        M_PI * M_PI * pow(D11 * D22 * D22 * D22, 0.25) / s_p / s_p;
    TacsScalar* Xtest = new TacsScalar[this->getShearGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = 0.0;  // log(1+gamma) = 0 since gamma=0 for unstiffened panel
    Xtest[3] = log(zeta);
    TacsScalar arg = this->getShearGP()->predictMeanTestData(Xtest);
    TacsScalar nondim_factor = exp(arg);
    TacsScalar output = dim_factor * nondim_factor;

    // backwards propagate sensitivities out of the axialGP model to nondim
    // params, (this part differentiates the nondim_factor)
    int n_shear_param = this->getShearGP()->getNparam();
    TacsScalar* Xtestsens = new TacsScalar[n_shear_param];
    TacsScalar Ysens = N12sens * output;
    this->getShearGP()->predictMeanTestDataSens(Ysens, Xtest, Xtestsens);

    *xisens += Xtestsens[0] / xi;
    *rho_0sens +=
        Xtestsens[1] / rho_0;  // chain rule dlog(rho_0) / drho_0 = 1/rho_0
    *zetasens += Xtestsens[3] / zeta;

    // backpropagate the dimensional factor terms out to the material and
    // geometric DVs, (this part is differentiating dim_factor)
    *D11sens += Ysens * 0.25 / D11;
    *D22sens += Ysens * 0.75 / D22;
    *spitchsens += Ysens * -2.0 / s_p;

    // return the final forward analysis output
    return output;

  } else {
    // use the CPT closed-form solution to compute the critical global axial
    // load no mode switching in this solution.. (only accurate for higher
    // aspect ratios => hence the need for machine learning for the actual
    // solution)
    TacsScalar lam1, lam2;  // lam1bar, lam2bar values
    TacsScalar dl1xi, dl2xi, _dl1gamma,
        _dl2gamma;  // gamma derivatives are private since unused here (gamma=0
                    // input)

    // compute the derivatives of the nondimensional constraints
    nondimShearParamsSens(xi, 0.0, &lam1, &lam2, &dl1xi, &_dl1gamma, &dl2xi,
                          &_dl2gamma);

    // compute forward analysis states involved in the N12crit load
    TacsScalar dim_factor = M_PI * M_PI * pow(D11 * D22 * D22 * D22, 0.25) /
                            this->stiffenerPitch / this->stiffenerPitch;
    TacsScalar num = 1.0 + pow(lam1, 4.0) + 6.0 * pow(lam1 * lam2, 2.0) +
                     pow(lam2, 4.0) + 2.0 * xi;
    TacsScalar den = 2.0 * lam1 * lam1 * lam2;
    TacsScalar nondim_factor = num / den;
    TacsScalar N12crit = dim_factor * nondim_factor;

    // sensitivities for the non_dim factor

    TacsScalar dNDlam1 =
        (4.0 * pow(lam1, 3.0) + 12.0 * lam1 * lam2 * lam2) / den -
        num * 4.0 * lam1 * lam2 / den / den;
    TacsScalar dNDlam2 =
        (4.0 * pow(lam2, 3.0) + 12.0 * lam2 * lam1 * lam1) / den -
        num * 2.0 * lam1 * lam1 / den / den;

    // compute the overall sensitivities
    *D11sens += N12sens * N12crit * 0.25 / D11;
    *D22sens += N12sens * N12crit * 0.75 / D22;
    *spitchsens += N12sens * N12crit * -2.0 / this->stiffenerPitch;
    *xisens +=
        N12sens * dim_factor * (dNDlam1 * dl1xi + dNDlam2 * dl2xi + 2.0 / den);

    // return N12crit from closed-form solution
    return N12crit;
  }
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::computeStiffenerCripplingLoad(
    const TacsScalar D11, const TacsScalar D22, const TacsScalar xi,
    const TacsScalar rho_0, const TacsScalar genPoiss, const TacsScalar zeta) {
  if (this->getCripplingGP()) {
    // use Gaussian processes to compute the critical global axial load
    TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                            this->stiffenerHeight / this->stiffenerHeight;
    TacsScalar* Xtest = new TacsScalar[this->getCripplingGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = log(genPoiss);
    Xtest[3] = log(zeta);
    return dim_factor * exp(this->getCripplingGP()->predictMeanTestData(Xtest));

  } else {
    // use the literature CPT closed-form solution for approximate stiffener
    // crippling, not a function of aspect ratio
    TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                            this->stiffenerHeight / this->stiffenerHeight;
    TacsScalar nondim_factor = (0.476 - 0.56 * (genPoiss - 0.2)) * xi;
    return dim_factor * nondim_factor;
  }
}

TacsScalar
TACSGPBladeStiffenedShellConstitutive::computeStiffenerCripplingLoadSens(
    const TacsScalar N1sens, const TacsScalar D11, const TacsScalar D22,
    const TacsScalar xi, const TacsScalar rho_0, const TacsScalar genPoiss,
    const TacsScalar zeta, TacsScalar* D11sens, TacsScalar* D22sens,
    TacsScalar* sheightsens, TacsScalar* xisens, TacsScalar* rho_0sens,
    TacsScalar* genPoiss_sens, TacsScalar* zetasens) {
  if (this->getCripplingGP()) {
    // use Gaussian processes to compute the critical global axial load
    TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                            this->stiffenerHeight / this->stiffenerHeight;
    TacsScalar* Xtest = new TacsScalar[this->getCripplingGP()->getNparam()];
    Xtest[0] = log(xi);
    Xtest[1] = log(rho_0);
    Xtest[2] = log(genPoiss);
    Xtest[3] = log(zeta);
    TacsScalar arg = this->getCripplingGP()->predictMeanTestData(Xtest);
    TacsScalar nondim_factor = exp(arg);
    TacsScalar output = dim_factor * nondim_factor;

    // backwards propagate sensitivities out of the crippling model to nondim
    // params, (this part differentiates the nondim_factor)
    int n_crippling_param = this->getCripplingGP()->getNparam();
    TacsScalar* Xtestsens = new TacsScalar[n_crippling_param];
    TacsScalar Ysens = N1sens * output;
    this->getCripplingGP()->predictMeanTestDataSens(Ysens, Xtest, Xtestsens);

    *xisens += Xtestsens[0] / xi;
    *rho_0sens +=
        Xtestsens[1] / rho_0;  // chain rule dlog(rho_0) / drho_0 = 1/rho_0
    *genPoiss_sens += Xtestsens[2] / genPoiss;
    *zetasens += Xtestsens[3] / zeta;

    // backpropagate the dimensional factor terms out to the material and
    // geometric DVs, (this part is differentiating the dimensional factor)
    *D11sens += Ysens * 0.5 / D11;
    *D22sens += Ysens * 0.5 / D22;
    *sheightsens += Ysens * -2.0 / this->stiffenerHeight;

    // return the final forward analysis output
    return output;

  } else {
    // use the literature CPT closed-form solution for approximate stiffener
    // crippling, not a function of aspect ratio
    TacsScalar dim_factor = M_PI * M_PI * sqrt(D11 * D22) /
                            this->stiffenerHeight / this->stiffenerHeight;
    TacsScalar nondim_factor = (0.476 - 0.56 * (genPoiss - 0.2)) * xi;
    TacsScalar N11crit = dim_factor * nondim_factor;

    // compute the derivatives
    TacsScalar outputsens = N1sens * N11crit;
    *D11sens += outputsens * 0.5 / D11;
    *D22sens += outputsens * 0.5 / D22;
    *xisens += outputsens / xi;
    *genPoiss_sens += outputsens / nondim_factor * -0.56 * xi;
    *sheightsens += outputsens * -2.0 / this->stiffenerHeight;

    // output the critical load here
    return N11crit;
  }
}

// -----------------------------------------------------------
//               DERIVATIVE TESTING SUBROUTINES
// -----------------------------------------------------------
// -----------------------------------------------------------

TacsScalar TACSGPBladeStiffenedShellConstitutive::testAffineAspectRatio(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 4;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.341;  // D11
  x0[1] = 5.216;   // D22
  x0[2] = 3.124;   // a
  x0[3] = 1.061;   // b

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  f0 = computeAffineAspectRatio(x[0], x[1], x[2], x[3]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  f2 = computeAffineAspectRatio(x[0], x[1], x[2], x[3]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  computeAffineAspectRatioSens(p_output, x0[0], x0[1], x0[2], x0[3],
                               &input_sens[0], &input_sens[1], &input_sens[2],
                               &input_sens[3]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testAffineAspectRatio:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testGeneralizedRigidity(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 4;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.341;  // D11
  x0[1] = 5.216;   // D22
  x0[2] = 6.132;   // D12
  x0[3] = 2.103;   // D66

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  f0 = computeGeneralizedRigidity(x[0], x[1], x[2], x[3]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  f2 = computeGeneralizedRigidity(x[0], x[1], x[2], x[3]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  computeGeneralizedRigiditySens(p_output, x0[0], x0[1], x0[2], x0[3],
                                 &input_sens[0], &input_sens[1], &input_sens[2],
                                 &input_sens[3]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testGeneralizedRigidity:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testGeneralizedPoissonsRatio(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 2;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.341;  // D12
  x0[1] = 5.381;   // D66

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  f0 = computeGeneralizedPoissonsRatio(x[0], x[1]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  f2 = computeGeneralizedPoissonsRatio(x[0], x[1]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  computeGeneralizedPoissonsRatioSens(p_output, x0[0], x0[1], &input_sens[0],
                                      &input_sens[1]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testGeneralizedPoissonsRatio:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testStiffenerAreaRatio(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 4;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = this->stiffenerThick;
  x0[1] = this->stiffenerHeight;
  x0[2] = this->stiffenerPitch;
  x0[3] = this->panelThick;

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;
  this->stiffenerThick -= p_input[0] * epsilon;
  this->stiffenerHeight -= p_input[1] * epsilon;
  this->stiffenerPitch -= p_input[2] * epsilon;
  this->panelThick -= p_input[3] * epsilon;
  f0 = computeStiffenerAreaRatio();

  this->stiffenerThick += 2.0 * p_input[0] * epsilon;
  this->stiffenerHeight += 2.0 * p_input[1] * epsilon;
  this->stiffenerPitch += 2.0 * p_input[2] * epsilon;
  this->panelThick += 2.0 * p_input[3] * epsilon;
  f2 = computeStiffenerAreaRatio();

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // reset the values
  this->stiffenerThick -= p_input[0] * epsilon;
  this->stiffenerHeight -= p_input[1] * epsilon;
  this->stiffenerPitch -= p_input[2] * epsilon;
  this->panelThick -= p_input[3] * epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  computeStiffenerAreaRatioSens(p_output, &input_sens[0], &input_sens[1],
                                &input_sens[2], &input_sens[3]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testStiffenerAreaRatio:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testStiffenerStiffnessRatio(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 4;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.2143;  // D11
  x0[1] = this->stiffenerThick;
  x0[2] = this->stiffenerHeight;
  x0[3] = this->stiffenerPitch;

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;
  TacsScalar D11 = x0[0] * 1.0;
  D11 -= p_input[0] * epsilon;
  this->stiffenerThick -= p_input[1] * epsilon;
  this->stiffenerHeight -= p_input[2] * epsilon;
  this->stiffenerPitch -= p_input[3] * epsilon;
  f0 = computeStiffenerStiffnessRatio(D11);

  D11 += 2.0 * p_input[0] * epsilon;
  this->stiffenerThick += 2.0 * p_input[1] * epsilon;
  this->stiffenerHeight += 2.0 * p_input[2] * epsilon;
  this->stiffenerPitch += 2.0 * p_input[3] * epsilon;
  f2 = computeStiffenerStiffnessRatio(D11);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // reset the values
  D11 -= p_input[0] * epsilon;
  this->stiffenerThick -= p_input[1] * epsilon;
  this->stiffenerHeight -= p_input[2] * epsilon;
  this->stiffenerPitch -= p_input[3] * epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  computeStiffenerStiffnessRatioSens(p_output, D11, &input_sens[0],
                                     &input_sens[1], &input_sens[2],
                                     &input_sens[3]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testStiffenerStiffnessRatio:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testTransverseShearParameter(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 4;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 100.234;  // A66
  x0[1] = 421.341;  // A11
  x0[2] = 2.134;    // b
  x0[3] = 0.0112;   // h

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  f0 = computeTransverseShearParameter(x[0], x[1], x[2], x[3]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  f2 = computeTransverseShearParameter(x[0], x[1], x[2], x[3]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  computeTransverseShearParameterSens(p_output, x0[0], x0[1], x0[2], x0[3],
                                      &input_sens[0], &input_sens[1],
                                      &input_sens[2], &input_sens[3]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testTransverseShearParameter:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testNondimensionalParameters(
    TacsScalar epsilon) {
  // run each of the nondim parameter tests and aggregate the max among them
  const int n_tests = 6;
  TacsScalar* relErrors = new TacsScalar[n_tests];

  relErrors[0] = testAffineAspectRatio(epsilon);
  relErrors[1] = testGeneralizedRigidity(epsilon);
  relErrors[2] = testGeneralizedPoissonsRatio(epsilon);
  relErrors[3] = testStiffenerAreaRatio(epsilon);
  relErrors[4] = testStiffenerStiffnessRatio(epsilon);
  relErrors[5] = testTransverseShearParameter(epsilon);

  // get max rel error among them
  TacsScalar maxRelError = 0.0;
  for (int i = 0; i < n_tests; i++) {
    if (relErrors[i] > maxRelError) {
      maxRelError = relErrors[i];
    }
  }

  // report the overall test results
  printf(
      "\n\nTACSGPBladeStiffened..testNondimensionalParmeters full results::\n");
  printf("\ttestAffineAspectRatio = %.4e\n", relErrors[0]);
  printf("\ttestGeneralizedRigidity = %.4e\n", relErrors[1]);
  printf("\ttestGeneralizedPoissonsRatio = %.4e\n", relErrors[2]);
  printf("\ttestStiffenerAreaRatio = %.4e\n", relErrors[3]);
  printf("\ttestStiffenerStiffnessRatio = %.4e\n", relErrors[4]);
  printf("\ttestTransverseShearParameter = %.4e\n", relErrors[5]);
  printf("\tOverall max rel error = %.4e\n", maxRelError);

  return maxRelError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testAxialCriticalLoads(
    TacsScalar epsilon) {
  // run each of the nondim parameter tests and aggregate the max among them
  const int n_tests = 2;
  TacsScalar* relErrors = new TacsScalar[n_tests];

  relErrors[0] = testCriticalGlobalAxialLoad(epsilon);
  relErrors[1] = testCriticalLocalAxialLoad(epsilon);

  // get max rel error among them
  TacsScalar maxRelError = 0.0;
  for (int i = 0; i < n_tests; i++) {
    if (relErrors[i] > maxRelError) {
      maxRelError = relErrors[i];
    }
  }

  // get max rel error among them
  printf("\n\nTACSGPBladeStiffened..testAxialCriticalLoads full results::\n");
  printf("\ttestGlobalAxialLoad = %.4e\n", relErrors[0]);
  printf("\ttestLocalAxialLoad = %.4e\n", relErrors[1]);
  printf("\tOverall max rel error = %.4e\n", maxRelError);

  return maxRelError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testCriticalGlobalAxialLoad(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 8;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.2412;  // D11
  x0[1] = 5.4323;   // D22
  x0[2] = 2.134;    // b
  x0[3] = 0.13432;  // delta
  x0[4] = 2.4545;   // rho0
  x0[5] = 1.24332;  // xi
  x0[6] = 0.2454;   // gamma
  x0[7] = 40.1324;  // zeta

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  f0 = computeCriticalGlobalAxialLoad(x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                                      x[7]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  f2 = computeCriticalGlobalAxialLoad(x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                                      x[7]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i];
  }
  computeCriticalGlobalAxialLoadSens(
      p_output, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], &input_sens[0],
      &input_sens[1], &input_sens[2], &input_sens[3], &input_sens[4],
      &input_sens[5], &input_sens[6], &input_sens[7]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testCriticalGlobalAxialLoad:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testCriticalLocalAxialLoad(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 6;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.2412;               // D11
  x0[1] = 5.4323;                // D22
  x0[2] = this->stiffenerPitch;  // s_p
  x0[3] = 2.4545;                // rho0
  x0[4] = 1.24332;               // xi
  x0[5] = 40.1324;               // zeta

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  this->stiffenerPitch = x0[2] - p_input[2] * epsilon;
  f0 = computeCriticalLocalAxialLoad(x[0], x[1], x[3], x[4], x[5]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  this->stiffenerPitch = x0[2] + p_input[2] * epsilon;
  f2 = computeCriticalLocalAxialLoad(x[0], x[1], x[3], x[4], x[5]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  this->stiffenerPitch = x0[2];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i];
  }
  computeCriticalLocalAxialLoadSens(
      p_output, x[0], x[1], x[3], x[4], x[5], &input_sens[0], &input_sens[1],
      &input_sens[2], &input_sens[3], &input_sens[4], &input_sens[5]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testCriticalLocalAxialLoad:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testShearCriticalLoads(
    TacsScalar epsilon) {
  // run each of the nondim parameter tests and aggregate the max among them
  const int n_tests = 2;
  TacsScalar* relErrors = new TacsScalar[n_tests];

  relErrors[0] = testCriticalGlobalShearLoad(epsilon);
  relErrors[1] = testCriticalLocalShearLoad(epsilon);

  // get max rel error among them
  TacsScalar maxRelError = 0.0;
  for (int i = 0; i < n_tests; i++) {
    if (relErrors[i] > maxRelError) {
      maxRelError = relErrors[i];
    }
  }

  // get max rel error among them
  printf("\n\nTACSGPBladeStiffened..testShearCriticalLoads full results::\n");
  printf("\ttestGlobalShearLoad = %.4e\n", relErrors[0]);
  printf("\ttestLocalShearLoad = %.4e\n", relErrors[1]);
  printf("\tOverall max rel error = %.4e\n", maxRelError);

  return maxRelError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testCriticalGlobalShearLoad(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 7;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.2412;  // D11
  x0[1] = 5.4323;   // D22
  x0[2] = 2.134;    // b
  x0[3] = 2.4545;   // rho0
  x0[4] = 1.24332;  // xi
  x0[5] = 0.2454;   // gamma
  x0[6] = 40.1324;  // zeta

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  f0 = computeCriticalGlobalShearLoad(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  f2 = computeCriticalGlobalShearLoad(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i];
  }
  computeCriticalGlobalShearLoadSens(
      p_output, x[0], x[1], x[2], x[3], x[4], x[5], x[6], &input_sens[0],
      &input_sens[1], &input_sens[2], &input_sens[3], &input_sens[4],
      &input_sens[5], &input_sens[6]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testCriticalGlobalShearLoad:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testCriticalLocalShearLoad(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 6;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.2412;               // D11
  x0[1] = 5.4323;                // D22
  x0[2] = this->stiffenerPitch;  // s_p
  x0[3] = 1.24332;               // xi
  x0[4] = 2.4545;                // rho0
  x0[5] = 40.1324;               // zeta

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  this->stiffenerPitch = x0[2] - p_input[2] * epsilon;
  f0 = computeCriticalLocalShearLoad(x[0], x[1], x[3], x[4], x[5]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  this->stiffenerPitch = x0[2] + p_input[2] * epsilon;
  f2 = computeCriticalLocalShearLoad(x[0], x[1], x[3], x[4], x[5]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  this->stiffenerPitch = x0[2];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i];
  }
  computeCriticalLocalShearLoadSens(
      p_output, x[0], x[1], x[3], x[4], x[5], &input_sens[0], &input_sens[1],
      &input_sens[2], &input_sens[3], &input_sens[4], &input_sens[5]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testCriticalLocalShearLoad:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testStiffenerCripplingLoad(
    TacsScalar epsilon) {
  // perform complex-step or finite difference check (depending on the value of
  // _eps/epsilon) generate random input perturbation and output perturbation
  // test vectors
  const int n_input = 7;
  TacsScalar* p_input = new TacsScalar[n_input];
  for (int ii = 0; ii < n_input; ii++) {
    p_input[ii] = ((double)rand() / (RAND_MAX));
  }
  TacsScalar p_output = ((double)rand() / (RAND_MAX));

  // compute initial values
  TacsScalar* x0 = new TacsScalar[n_input];
  x0[0] = 10.2412;                // D11
  x0[1] = 5.4323;                 // D22
  x0[2] = 1.24332;                // xi
  x0[3] = 2.4545;                 // rho0
  x0[4] = 0.2454;                 // genPoiss
  x0[5] = 40.1324;                // zeta
  x0[6] = this->stiffenerHeight;  // sheight

  // perform central difference over rho_0 function on [D11,D22,a,b]
  TacsScalar f0, f1, f2;

  TacsScalar* x = new TacsScalar[n_input];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] - p_input[i] * epsilon;
  }
  this->stiffenerHeight = x0[6] - p_input[6] * epsilon;
  f0 = computeStiffenerCripplingLoad(x[0], x[1], x[2], x[3], x[4], x[5]);

  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i] + p_input[i] * epsilon;
  }
  this->stiffenerHeight = x0[6] + p_input[6] * epsilon;
  f2 = computeStiffenerCripplingLoad(x[0], x[1], x[2], x[3], x[4], x[5]);

  TacsScalar centralDiff = (f2 - f0) / 2.0 / epsilon;

  // now perform the adjoint sensitivity
  TacsScalar* input_sens = new TacsScalar[n_input];
  this->stiffenerHeight = x0[6];
  for (int i = 0; i < n_input; i++) {
    x[i] = x0[i];
  }
  computeStiffenerCripplingLoadSens(
      p_output, x[0], x[1], x[2], x[3], x[4], x[5], &input_sens[0],
      &input_sens[1], &input_sens[2], &input_sens[3], &input_sens[4],
      &input_sens[5], &input_sens[6]);
  TacsScalar adjTD = 0.0;
  for (int j = 0; j < n_input; j++) {
    adjTD += input_sens[j] * p_input[j];
  }
  adjTD = TacsRealPart(adjTD);

  // compute relative error
  TacsScalar relError = abs((adjTD - centralDiff) / centralDiff);
  printf("TACSGPBladeStiffened..testStiffenerCripplingLoad:\n");
  printf("\t adjDeriv = %.4e\n", adjTD);
  printf("\t centralDiff = %.4e\n", centralDiff);
  printf("\t rel error = %.4e\n", relError);
  return relError;
}

TacsScalar TACSGPBladeStiffenedShellConstitutive::testAllTests(
    TacsScalar epsilon) {
  // run each of the nondim parameter tests and aggregate the max among them
  const int n_tests = 7;
  TacsScalar* relErrors = new TacsScalar[n_tests];

  relErrors[0] = testNondimensionalParameters(epsilon);
  relErrors[1] = testAxialCriticalLoads(epsilon);
  relErrors[2] = testShearCriticalLoads(epsilon);
  relErrors[3] = testStiffenerCripplingLoad(epsilon);
  relErrors[4] = 0.0;
  relErrors[5] = 0.0;
  relErrors[6] = 0.0;
  if (this->getAxialGP()) {
    relErrors[4] = this->getAxialGP()->testAllGPTests(epsilon);
  }
  if (this->getShearGP()) {
    relErrors[5] = this->getShearGP()->testAllGPTests(epsilon);
  }
  if (this->getCripplingGP()) {
    relErrors[6] = this->getCripplingGP()->testAllGPTests(epsilon);
  }

  // get max rel error among them
  TacsScalar maxRelError = 0.0;
  for (int i = 0; i < n_tests; i++) {
    if (relErrors[i] > maxRelError) {
      maxRelError = relErrors[i];
    }
  }

  // get max rel error among them
  printf("\n\nTACSGPBladeStiffened..testAllTests full results::\n");
  printf("\ttestNondimensionalParameters = %.4e\n", relErrors[0]);
  printf("\ttestAxialCriticalLoads = %.4e\n", relErrors[1]);
  printf("\ttestShearCriticalLoads = %.4e\n", relErrors[2]);
  printf("\ttestStiffenerCripplingLoad = %.4e\n", relErrors[3]);
  if (this->getAxialGP()) {
    printf("\ttestAxialGP all tests = %.4e", relErrors[4]);
  }
  if (this->getShearGP()) {
    printf("\ttestShearGP all tests = %.4e", relErrors[5]);
  }
  if (this->getCripplingGP()) {
    printf("\ttestCripplingGp all tests = %.4e", relErrors[6]);
  }
  printf("\tOverall max rel error = %.4e\n", maxRelError);

  return maxRelError;
}