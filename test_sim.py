def test_sim():
    import os
    from sys import stdout

    from simtk.openmm import LangevinMiddleIntegrator, Platform
    from simtk.openmm.app import (
        PME,
        ForceField,
        HBonds,
        PDBFile,
        Simulation,
        StateDataReporter,
    )
    from simtk.unit import kelvin, nanometer, picosecond, picoseconds

    pdb = PDBFile("input.pdb")
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * nanometer,
        constraints=HBonds,
    )
    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.004 * picoseconds
    )
    if os.getenv("TEST_MODE") == "CPU":
        print("Using CPU")
        platform = Platform.getPlatformByName("CPU")
        simulation = Simulation(pdb.topology, system, integrator, platform)
    else:
        print("Using GPU")
        platform = Platform.getPlatformByName("CUDA")
        simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    print("minimizing energy")
    simulation.minimizeEnergy()
    simulation.reporters.append(
        StateDataReporter(
            stdout, 1000, step=True, potentialEnergy=True, temperature=True
        )
    )
    simulation.step(10000)


if __name__ == "__main__":
    test_sim()
