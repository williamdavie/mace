import logging
from typing import Callable, Dict, Optional, Type

import torch

logging.getLogger("cuequivariance_torch.primitives.tensor_product").setLevel(logging.WARNING)
logging.getLogger("cuequivariance_torch.primitives.symmetric_tensor_product").setLevel(logging.WARNING)

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    GeneralNonLinearBiasReadoutBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearBiasReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
    MagneticRealAgnosticDensityInteractionBlock,
    MagneticRealAgnosticSeparateRadialDensityInteractionBlock,
    MagneticRealAgnosticSeparateRadialCoupledDensityInteractionBlock,
    MagneticRealAgnosticSeparateRadialCoupledPosToMagDensityInteractionBlock,
    MagneticRealAgnosticSeparateRadialDensityTestingInteractionBlock,
    MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock,
    MagneticRealAgnosticFlexibleSpinOrbitCoupledDensityInteractionBlock,
    MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock,
    MagneticRealAgnosticSpinOrbitCoupledMagmomDensityInteractionBlock,
    MagneticRealAgnosticResidueSpinOrbitCoupledMagmomDensityInteractionBlock,
    MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock,
    MagneticRealAgnosticSpinOrbitCoupledDensityWithMagmomInteractionBlock,
    MagneticRealAgnosticResidueSpinOrbitCoupledDensityWithMagmomInteractionBlock,
    ScaleShiftBlock,
)
from .extensions import PolarMACE
from .gate import GatedEquivariantBlock
from .loss import (
    DipolePolarLoss,
    DipoleSingleLoss,
    UniversalLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesL1L2Loss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
    EvenSpline1BodyLoss,
)
from .models import (
    MACE,
    AtomicDielectricMACE,
    AtomicDipolesMACE,
    EnergyDipolesMACE,
    ScaleShiftMACE,
    #
    MagneticSCFMACE,
    MagneticScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledScaleShiftMACE,
    MagneticSolidHarmonicsScaleShiftMACE,
    MagneticSolidHarmonicsSeparateReadoutScaleShiftMACE,
    MagneticSolidHarmonicsSeparateReadoutMixMagmomScaleShiftMACE,
    MagneticSolidHarmonicsFlexibleSOScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledWithSelfMagmomScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledWithOneBodySelfMagmomScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledWithOneBodyReadoutSelfMagmomScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledWithOneBodyGinzburgSelfMagmomScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledWithOneBodyMultiSpeciesGinzburgSelfMagmomScaleShiftMACE,
    MagneticSolidHarmonicsNonSpinOrbitCoupledWithOneBodyMultiSpeciesGinzburgSelfMagmomScaleShiftMACE,
    MagneticSolidHarmonicsFixingNonSpinOrbitCoupledWithOneBodyMultiSpeciesGinzburgSelfMagmomScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledWithOneBodyMultiSpeciesFixingGinzburgSelfMagmomScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledWithSelfMagmomFixingScaleShiftMACE,
    MagneticSolidHarmonicsSpinOrbitCoupledWithOneBodyMultiSpeciesEvenSplineSelfMagmomScaleShiftMACE,
    #
    EvenMagSaturationBarrier
)

from .radial import BesselBasis, GaussianBasis, PolynomialCutoff, ZBLBasis
from .symmetric_contraction import SymmetricContraction, NonSOCSymmetricContraction
from .utils import (
    compute_avg_num_neighbors,
    compute_dielectric_gradients,
    compute_fixed_charge_dipole,
    compute_fixed_charge_dipole_polar,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_rms_dipoles,
    compute_statistics,
)

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
    "MagneticRealAgnosticDensityInteractionBlock": MagneticRealAgnosticDensityInteractionBlock,
    "MagneticRealAgnosticSeparateRadialDensityInteractionBlock": MagneticRealAgnosticSeparateRadialDensityInteractionBlock,
    "MagneticRealAgnosticSeparateRadialCoupledDensityInteractionBlock": MagneticRealAgnosticSeparateRadialCoupledDensityInteractionBlock,
    "MagneticRealAgnosticSeparateRadialCoupledPosToMagDensityInteractionBlock": MagneticRealAgnosticSeparateRadialCoupledPosToMagDensityInteractionBlock,
    "MagneticRealAgnosticSeparateRadialDensityTestingInteractionBlock": MagneticRealAgnosticSeparateRadialDensityTestingInteractionBlock,
    "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock": MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock,
    "MagneticRealAgnosticFlexibleSpinOrbitCoupledDensityInteractionBlock": MagneticRealAgnosticFlexibleSpinOrbitCoupledDensityInteractionBlock,
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock": MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock,
    "MagneticRealAgnosticSpinOrbitCoupledMagmomDensityInteractionBlock": MagneticRealAgnosticSpinOrbitCoupledMagmomDensityInteractionBlock,
    "MagneticRealAgnosticResidueSpinOrbitCoupledMagmomDensityInteractionBlock": MagneticRealAgnosticResidueSpinOrbitCoupledMagmomDensityInteractionBlock,
    "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock": MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock,
    "MagneticRealAgnosticSpinOrbitCoupledDensityWithMagmomInteractionBlock": MagneticRealAgnosticSpinOrbitCoupledDensityWithMagmomInteractionBlock,
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityWithMagmomInteractionBlock": MagneticRealAgnosticResidueSpinOrbitCoupledDensityWithMagmomInteractionBlock,

}


readout_classes: Dict[str, Type[LinearReadoutBlock]] = {
    "LinearReadoutBlock": LinearReadoutBlock,
    "LinearDipoleReadoutBlock": LinearDipoleReadoutBlock,
    "NonLinearDipoleReadoutBlock": NonLinearDipoleReadoutBlock,
    "NonLinearReadoutBlock": NonLinearReadoutBlock,
    "NonLinearBiasReadoutBlock": NonLinearBiasReadoutBlock,
    "GeneralNonLinearBiasReadoutBlock": GeneralNonLinearBiasReadoutBlock,
    "MagneticRealAgnosticDensityInteractionBlock": MagneticRealAgnosticDensityInteractionBlock,
    "MagneticRealAgnosticSeparateRadialDensityInteractionBlock": MagneticRealAgnosticSeparateRadialDensityInteractionBlock,
    "MagneticRealAgnosticSeparateRadialCoupledDensityInteractionBlock": MagneticRealAgnosticSeparateRadialCoupledDensityInteractionBlock,
    "MagneticRealAgnosticSeparateRadialCoupledPosToMagDensityInteractionBlock": MagneticRealAgnosticSeparateRadialCoupledPosToMagDensityInteractionBlock,
    "MagneticRealAgnosticSeparateRadialDensityTestingInteractionBlock": MagneticRealAgnosticSeparateRadialDensityTestingInteractionBlock,
    "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock": MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock,
    "MagneticRealAgnosticFlexibleSpinOrbitCoupledDensityInteractionBlock": MagneticRealAgnosticFlexibleSpinOrbitCoupledDensityInteractionBlock,
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock": MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock,
    "MagneticRealAgnosticSpinOrbitCoupledMagmomDensityInteractionBlock": MagneticRealAgnosticSpinOrbitCoupledMagmomDensityInteractionBlock,
    "MagneticRealAgnosticResidueSpinOrbitCoupledMagmomDensityInteractionBlock": MagneticRealAgnosticResidueSpinOrbitCoupledMagmomDensityInteractionBlock,
    "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock": MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock,
    "MagneticRealAgnosticSpinOrbitCoupledDensityWithMagmomInteractionBlock": MagneticRealAgnosticSpinOrbitCoupledDensityWithMagmomInteractionBlock,
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityWithMagmomInteractionBlock": MagneticRealAgnosticResidueSpinOrbitCoupledDensityWithMagmomInteractionBlock,

}

scaling_classes: Dict[str, Callable] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
    "rms_dipoles_scaling": compute_rms_dipoles,
}

gate_dict: Dict[str, Optional[Callable]] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

__all__ = [
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "ZBLBasis",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "LinearDipoleReadoutBlock",
    "LinearDipolePolarReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "NonLinearDipolePolarReadoutBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "GaussianBasis",
    "MACE",
    "ScaleShiftMACE",
    "AtomicDipolesMACE",
    "AtomicDielectricMACE",
    "EnergyDipolesMACE",
    "PolarMACE",
    "MagneticScaleShiftMACE",
    "MagneticSolidHarmonicsScaleShiftMACE",
    "MagneticSolidHarmonicsSpinOrbitCoupledScaleShiftMACE",
    "WeightedEnergyForcesLoss",
    "WeightedForcesLoss",
    "WeightedEnergyForcesVirialsLoss",
    "WeightedEnergyForcesStressLoss",
    "DipoleSingleLoss",
    "WeightedEnergyForcesDipoleLoss",
    "WeightedHuberEnergyForcesStressLoss",
    "UniversalLoss",
    "WeightedEnergyForcesL1L2Loss",
    "EvenSpline1BodyLoss",
    "SymmetricContraction",
    "NonSOCSymmetricContraction",
    "interaction_classes",
    "compute_mean_std_atomic_inter_energy",
    "compute_avg_num_neighbors",
    "compute_statistics",
    "compute_fixed_charge_dipole",
    "compute_fixed_charge_dipole_polar",
    "compute_dielectric_gradients",
]
