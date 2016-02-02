"""
Contains classes for parsing LAMMPS data and input files
"""
from parmed.utils.six import PY2

__all__ = ['LammpsDataFile', 'LammpsInputFile']

if PY2:
    def _which(program):
        import os
        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
        return None
else:
    from shutil import which as _which

LAMMPS_EXE = ''
# TODO: More robust way to check for lammps installation.
for exe in ['lammps', 'lmp_mpi', 'lmp_serial', 'lmp_openmpi', 'lmp_mac_mpi']:
    if _which(exe):
        LAMMPS_EXE = exe
        break

del _which

from parmed.lammps.lammpsdata import LammpsDataFile
from parmed.lammps.lammpsinput import LammpsInputFile
