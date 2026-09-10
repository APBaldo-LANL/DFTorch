"""PLUMED interface for PyTorch molecular dynamics simulations."""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import torch
import numpy as np
import os

if os.environ["PLUMED_KERNEL"]:
    try:
        import plumed
    except ImportError:
        warnings.warn(
            "PLUMED Python module not found. Install with: pip install plumed",
            ImportWarning,
        )


class PlumedInterface:
    """PyTorch-compatible interface to PLUMED for collective variable biasing.

    This class provides a bridge between PyTorch MD simulations and PLUMED,
    enabling free-energy calculations, metadynamics, and other enhanced
    sampling methods.

    Parameters
    ----------
    natoms : int
        Number of atoms in the system.
    timestep : float
        MD timestep in picoseconds (ps).
    plumed_file : str
        Path to the PLUMED input file defining collective variables and biases.
    log_file : str, optional
        Path to PLUMED log file. Default is 'plumed.log'.
    restart : bool, optional
        If True, restart from a previous PLUMED simulation. Default is False.
    
    Attributes
    ----------
    plumed : plumed.Plumed
        The PLUMED interface object.
    natoms : int
        Number of atoms.
    step : int
        Current simulation step.
    forces_buffer : np.ndarray
        Buffer for forces returned by PLUMED.
    virial_buffer : np.ndarray
        Buffer for virial tensor returned by PLUMED.
    
    Examples
    --------
    >>> plumed_interface = PlumedInterface(
    ...     natoms=100,
    ...     timestep=0.001,
    ...     plumed_file='plumed.dat'
    ... )
    >>> positions = torch.randn(100, 3, requires_grad=True)
    >>> cell = torch.eye(3) * 20.0
    >>> energy, forces = plumed_interface.calculate(positions, cell)
    """

    def __init__(
        self,
        natoms: int,
        timestep: float,
        plumed_file: str,
        log_file: str = "plumed.log",
        restart: bool = True,
    ) -> None:
        """Initialize PLUMED interface."""

        fs_to_ps = 1000 # unit conversion from DFTorch (fs) to plumed (ps)
        self.natoms = natoms
        self.timestep = timestep / fs_to_ps
        self.step = 0
        
        # Initialize PLUMED
        self.plumed = plumed.Plumed()
        
        # Set up PLUMED commands
        self.plumed.cmd("setNatoms", self.natoms)
        self.plumed.cmd("setLogFile", log_file)
        self.plumed.cmd("setTimestep", self.timestep)
        
        if restart:
            self.plumed.cmd("setRestart", 1)
        
        # Read PLUMED input file
        self.plumed.cmd("setPlumedDat", plumed_file)
        self.plumed.cmd("setMDEngine", "pytorch")
        
        # Initialize PLUMED
        self.plumed.cmd("init")
        
        # Create buffers for forces and virial
        self.forces_buffer = np.zeros((natoms, 3), dtype=np.float64)
        self.virial_buffer = np.zeros((3, 3), dtype=np.float64)
        
        print(f"PLUMED interface initialized with {natoms} atoms")
        print(f"  Input file: {plumed_file}")
        print(f"  Log file: {log_file}")
        print(f"  Timestep: {timestep} ps")

    def calculate(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        masses: Optional[torch.Tensor] = None,
        charges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate bias energy and forces from PLUMED collective variables.

        This function passes atomic coordinates (and optionally cell, masses, charges)
        to PLUMED, which evaluates the defined collective variables and returns the
        bias potential energy and forces.

        Parameters
        ----------
        positions : torch.Tensor
            Atomic positions, shape `(natoms, 3)`, in Angstroms.
            Should have `requires_grad=True` for force computation.
        cell : torch.Tensor, optional
            Simulation cell vectors, shape `(3, 3)`, in Angstroms.
            If provided, enables periodic boundary conditions in PLUMED.
        masses : torch.Tensor, optional
            Atomic masses, shape `(natoms,)`, in atomic mass units (amu).
        charges : torch.Tensor, optional
            Atomic charges, shape `(natoms,)`, in electron units.

        Returns
        -------
        energy : torch.Tensor
            Bias potential energy from PLUMED, scalar tensor in eV.
        forces : torch.Tensor
            Bias forces from PLUMED, shape `(natoms, 3)`, in eV/Angstrom.
            These should be *added* to MD forces (PLUMED returns -dV/dr).

        Notes
        -----
        - Forces are computed via PyTorch autograd when available, or directly
          from PLUMED's analytical derivatives.
        - Units: PLUMED internally uses kJ/mol; conversion factors are applied.
        - The step counter is automatically incremented.
        
        Raises
        ------
        ValueError
            If positions shape doesn't match initialized natoms.
        """
        
        if positions.shape[0] != self.natoms:
            raise ValueError(
                f"Position array has {positions.shape[0]} atoms, "
                f"but interface was initialized with {self.natoms}"
            )
        
        # Convert to numpy for PLUMED (detach from computation graph temporarily)
        pos_np = positions.detach().cpu().numpy().astype(np.float64)
        
        # Reset force buffer
        self.forces_buffer.fill(0.0)
        
        # Pass data to PLUMED
        self.plumed.cmd("setStep", self.step)
        self.plumed.cmd("setPositions", pos_np)
        self.plumed.cmd("setForces", self.forces_buffer)
        
        if cell is not None:
            cell_np = cell.detach().cpu().numpy().astype(np.float64)
            self.plumed.cmd("setBox", cell_np)
            self.plumed.cmd("setVirial", self.virial_buffer)
        
        if masses is not None:
            masses_np = masses.detach().cpu().numpy().astype(np.float64)
            self.plumed.cmd("setMasses", masses_np)
        
        if charges is not None:
            charges_np = charges.detach().cpu().numpy().astype(np.float64)
            self.plumed.cmd("setCharges", charges_np)
        
        # Calculate collective variables and bias
        self.plumed.cmd("calc")
        
        # Get bias energy from PLUMED (in kJ/mol)
        bias_energy = np.zeros(1, dtype=np.float64)
        self.plumed.cmd("getBias", bias_energy)
        
        # Convert to PyTorch tensors
        # PLUMED energy is in kJ/mol, convert to eV
        kj_to_ev = 0.01036427  # Conversion factor
        energy = torch.tensor(
            bias_energy[0] * kj_to_ev,
            dtype=positions.dtype,
            device=positions.device,
        )
        
        # PLUMED forces are in kJ/mol/Angstrom, convert to eV/Angstrom
        forces = torch.tensor(
            self.forces_buffer * kj_to_ev,
            dtype=positions.dtype,
            device=positions.device,
        )
        
        # Increment step counter
        self.step += 1
        
        return energy, forces

    def calculate_with_autograd(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        masses: Optional[torch.Tensor] = None,
        charges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate bias using PyTorch autograd for gradient computation.

        This is an alternative to :meth:`calculate` that uses PyTorch's automatic
        differentiation to compute forces, enabling full integration with
        gradient-based optimization and torch.compile.

        Parameters
        ----------
        Same as :meth:`calculate`.

        Returns
        -------
        energy : torch.Tensor
            Bias potential energy, scalar, differentiable.
        forces : torch.Tensor
            Bias forces computed via autograd as `-grad(energy, positions)`.

        Notes
        -----
        This method wraps the PLUMED calculation in a custom autograd Function,
        allowing gradients to flow through the collective variable bias.
        """
        
        return PlumedFunction.apply(
            positions, cell, masses, charges, self
        )

    def finalize(self) -> None:
        """Finalize PLUMED and write output files.
        
        Should be called at the end of the simulation to ensure all PLUMED
        output is properly flushed and finalized.
        """
        self.plumed.finalize()
        print(f"PLUMED finalized after {self.step} steps")

    def __del__(self) -> None:
        """Destructor ensures PLUMED is finalized."""
        if hasattr(self, 'plumed'):
            try:
                self.plumed.finalize()
            except:
                pass


class PlumedFunction(torch.autograd.Function):
    """Custom autograd function for PLUMED bias potential.
    
    This allows PyTorch's autograd to compute gradients through PLUMED's
    collective variable calculations, enabling full differentiability.
    """

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        masses: Optional[torch.Tensor],
        charges: Optional[torch.Tensor],
        plumed_interface: PlumedInterface,
    ) -> torch.Tensor:
        """Forward pass: compute PLUMED bias energy."""
        
        ctx.plumed_interface = plumed_interface
        ctx.save_for_backward(positions, cell, masses, charges)
        
        energy, _ = plumed_interface.calculate(
            positions, cell, masses, charges
        )
        
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: compute gradients using PLUMED forces."""
        
        positions, cell, masses, charges = ctx.saved_tensors
        plumed_interface = ctx.plumed_interface
        
        # Get forces from PLUMED (forces = -gradient)
        _, forces = plumed_interface.calculate(
            positions, cell, masses, charges
        )
        
        # Gradient w.r.t. positions: chain rule with upstream gradient
        grad_positions = -forces * grad_output
        
        # No gradients for other inputs
        return grad_positions, None, None, None, None


def initialize_plumed(
    natoms: int,
    timestep: float,
    plumed_input: str,
    log_file: str = "plumed.log",
    restart: bool = False,
    debug: bool = False,
) -> PlumedInterface:
    """Initialize PLUMED interface for PyTorch MD simulation.

    Convenience function to create and configure a PLUMED interface object.
    This should be called once at the start of the simulation.

    Parameters
    ----------
    natoms : int
        Total number of atoms in the system.
    timestep : float
        MD integration timestep in picoseconds (ps).
    plumed_input : str
        Path to PLUMED input file (typically 'plumed.dat') defining:
        - Collective variables (DISTANCE, ANGLE, TORSION, etc.)
        - Biasing potentials (METAD, RESTRAINT, UPPER_WALLS, etc.)
        - Output directives (PRINT, FLUSH)
    log_file : str, optional
        Path for PLUMED log output. Default is 'plumed.log'.
    restart : bool, optional
        Whether to restart from a previous PLUMED run. Default is False.
        If True, PLUMED will read HILLS files and restart metadynamics.
    debug : bool, optional
        Enable verbose debug output. Default is False.

    Returns
    -------
    PlumedInterface
        Initialized PLUMED interface ready for MD calculations.

    Examples
    --------
    Initialize PLUMED for a 100-atom system with 0.5 fs timestep:

    >>> plumed = initialize_plumed(
    ...     natoms=100,
    ...     timestep=0.0005,  # 0.5 fs in ps
    ...     plumed_input='plumed.dat',
    ... )

    Example PLUMED input file ('plumed.dat'):
    
    ```
    # Define distance collective variable
    d: DISTANCE ATOMS=1,2
    
    # Apply harmonic restraint
    RESTRAINT ARG=d AT=2.5 KAPPA=100.0
    
    # Print CV to file every 100 steps
    PRINT ARG=d FILE=colvar.dat STRIDE=100
    ```

    Notes
    -----
    - PLUMED uses atomic units internally but this interface handles conversions
    - Supported CV types: distances, angles, torsions, coordination numbers,
      path collective variables, and many more
    - See PLUMED documentation: https://www.plumed.org/
    """
    
    interface = PlumedInterface(
        natoms=natoms,
        timestep=timestep,
        plumed_file=plumed_input,
        log_file=log_file,
        restart=restart,
    )
    
    if debug:
        print("PLUMED interface created successfully")
        print(f"  Device: CPU (PLUMED operates on CPU)")
        print(f"  Step counter initialized to: {interface.step}")
    
    return interface


def compute_plumed_bias(
    plumed_interface: PlumedInterface,
    positions: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    masses: Optional[torch.Tensor] = None,
    charges: Optional[torch.Tensor] = None,
    use_autograd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PLUMED bias energy and forces for given atomic configuration.

    This is the main calculation function that should be called at each MD step
    to evaluate collective variables and apply biasing forces.

    Parameters
    ----------
    plumed_interface : PlumedInterface
        Initialized PLUMED interface object from :func:`initialize_plumed`.
    positions : torch.Tensor
        Atomic Cartesian coordinates, shape `(natoms, 3)`, in Angstroms.
        Should have `requires_grad=True` if computing forces via autograd.
    cell : torch.Tensor, optional
        Periodic cell matrix, shape `(3, 3)`, in Angstroms.
        Rows are lattice vectors: cell[0] = a, cell[1] = b, cell[2] = c.
        Required for CVs that use periodic boundary conditions.
    masses : torch.Tensor, optional
        Atomic masses, shape `(natoms,)`, in atomic mass units (amu).
        Required for mass-weighted CVs (e.g., center of mass).
    charges : torch.Tensor, optional
        Atomic charges, shape `(natoms,)`, in elementary charge units.
        Required for charge-dependent CVs.
    use_autograd : bool, optional
        If True, compute forces using PyTorch autograd. If False (default),
        use PLUMED's analytical derivatives. Default is False.

    Returns
    -------
    bias_energy : torch.Tensor
        Bias potential energy from collective variables, scalar, in eV.
        Add this to the MD potential energy.
    bias_forces : torch.Tensor
        Bias forces, shape `(natoms, 3)`, in eV/Angstrom.
        Add these to the MD forces (PLUMED convention: F = -dV/dr).

    Examples
    --------
    Basic usage in an MD loop:

    >>> for step in range(nsteps):
    ...     # ... compute MD forces and energy ...
    ...     
    ...     # Add PLUMED bias
    ...     plumed_energy, plumed_forces = compute_plumed_bias(
    ...         plumed_interface, positions, cell
    ...     )
    ...     
    ...     total_energy = md_energy + plumed_energy
    ...     total_forces = md_forces + plumed_forces
    ...     
    ...     # ... integrate equations of motion ...

    Using autograd for gradient-based optimization:

    >>> positions.requires_grad_(True)
    >>> energy, forces = compute_plumed_bias(
    ...     plumed_interface, positions, use_autograd=True
    ... )
    >>> energy.backward()  # Gradients in positions.grad

    Notes
    -----
    - Units: positions in Å, energy in eV, forces in eV/Å
    - PLUMED step counter is automatically incremented
    - For metadynamics, Gaussian hills are added according to PLUMED input
    - The bias should be added to (not subtracted from) the MD forces
    """
    
    if use_autograd:
        # Use custom autograd function for full differentiability
        energy = PlumedFunction.apply(
            positions, cell, masses, charges, plumed_interface
        )
        forces = -torch.autograd.grad(
            energy,
            positions,
            create_graph=True,
            retain_graph=True,
        )[0]
        return energy, forces
    else:
        # Use PLUMED's analytical forces directly (more efficient)
        return plumed_interface.calculate(
            positions, cell, masses, charges
        )
