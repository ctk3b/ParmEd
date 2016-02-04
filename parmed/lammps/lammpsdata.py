"""
This module contains functionality relevant to loading a GROMACS topology file
and building a Structure from it
"""
from __future__ import print_function, division, absolute_import

from collections import OrderedDict, defaultdict
from contextlib import closing
import copy
from datetime import datetime
import math
import os
import re
try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters
import sys
import warnings

from parmed.constants import TINY, DEG_TO_RAD
from parmed.exceptions import LammpsError, LammpsWarning, ParameterError
from parmed.formats.registry import FileFormatType
from parmed.parameters import ParameterSet
from parmed.structure import Structure
from parmed.topologyobjects import (Atom, Bond, Angle, Dihedral, Improper,
            NonbondedException, ExtraPoint, BondType, Cmap, NoUreyBradley,
            AngleType, DihedralType, DihedralTypeList, ImproperType, CmapType,
            RBTorsionType, ThreeParticleExtraPointFrame, AtomType, UreyBradley,
            TwoParticleExtraPointFrame, OutOfPlaneExtraPointFrame,
            NonbondedExceptionType, UnassignedAtomType)
from parmed.periodic_table import element_by_mass, AtomicNum
from parmed import unit as u
from parmed.utils.io import genopen
from parmed.utils.six import add_metaclass, string_types, iteritems
from parmed.utils.six.moves import range


# Lammps uses "funct" flags in its parameter files to indicate what kind of
# functional form is used for each of its different parameter types. This is
# taken from the topdirs.c source code file along with a table in the Lammps
# user manual. The table below summarizes my findings, for reference:

# Bonds
# -----
#  1 - F_BONDS : simple harmonic potential
#  2 - F_G96BONDS : fourth-power potential
#  3 - F_MORSE : morse potential
#  4 - F_CUBICBONDS : cubic potential
#  5 - F_CONNBONDS : not even implemented in GROMACS
#  6 - F_HARMONIC : seems to be the same as (1) ??
#  7 - F_FENEBONDS : finietely-extensible-nonlinear-elastic (FENE) potential
#  8 - F_TABBONDS : bond function from tabulated function
#  9 - F_TABBONDSNC : bond function from tabulated function (no exclusions)
# 10 - F_RESTRBONDS : restraint bonds

# Angles
# ------
#  1 - F_ANGLES : simple harmonic potential
#  2 - F_G96ANGLES : cosine-based angle potential
#  3 - F_CROSS_BOND_BONDS : bond-bond cross term potential
#  4 - F_CROSS_BOND_ANGLES : bond-angle cross term potential
#  5 - F_UREY_BRADLEY : Urey-Bradley angle-bond potential
#  6 - F_QUARTIC_ANGLES : 4th-order polynomial potential
#  7 - F_TABANGLES : angle function from tabulated function
#  8 - F_LINEAR_ANGLES : angle function from tabulated function
#  9 - F_RESTRANGLES : restricted bending potential

# Dihedrals
# ---------
#  1 - F_PDIHS : periodic proper torsion potential [ k(1+cos(n*phi-phase)) ]
#  2 - F_IDIHS : harmonic improper torsion potential
#  3 - F_RBDIHS : Ryckaert-Bellemans torsion potential
#  4 - F_PIDIHS : periodic harmonic improper torsion potential (same as 1)
#  5 - F_FOURDIHS : Fourier dihedral torsion potential
#  8 - F_TABDIHS : dihedral potential from tabulated function
#  9 - F_PDIHS : Same as 1, but can be multi-term
# 10 - F_RESTRDIHS : Restricted torsion potential
# 11 - F_CBTDIHS : combined bending-torsion potential


@add_metaclass(FileFormatType)
class LammpsDataFile(Structure):
    """ Class providing a parser and writer for a LAMMPS data file

    Parameters
    ----------
    fname : str
        The name of the file to read
    parametrized : bool, optional
        If True, parameters are assigned after parsing is done from the
        parametertypes sections. If False, only parameter types defined in the
        parameter sections themselves are loaded (i.e., on the same line as the
        parameter was defined). Default is True

    """

    #===================================================

    @staticmethod
    def id_format(filename):
        """ Identifies the file as a LAMMPS data file

        Parameters
        ----------
        filename : str
            Name of the file to check if it is a LAMMPS data file

        Returns
        -------
        is_fmt : bool
            If it is identified as a LAMMPS data file, return True. False
            otherwise
        """
        with closing(genopen(filename)) as f:
            f.readline()  # Title
            f.readline()  # Blank
            line = f.readline().split()
            if line[1] != 'atoms':  # 3rd line must be "N atoms".
                return False
            required_sections_found = 0
            previous_line = line
            for line in f:
                line = line.partition('#')[0].strip()  # Strip comments.
                if line in ('Masses', 'Atoms'):  # The two required sections.
                    next_line = next(f).partition('#')[0].strip()
                    if not next_line and not previous_line:
                        required_sections_found += 1
                previous_line = line
            if required_sections_found == 2:
                return True

            return False

    def set_units(self, unit_set):
        """Set what unit set to use. """
        self.RAD = u.radians
        self.DEGREE = u.degrees
        self.MOLE = u.mole
        self.TEMP = u.kelvin
        if unit_set == 'real':
            self.DIST = u.angstroms
            self.VEL = u.angstroms / u.femtosecond
            self.ENERGY = u.kilocalorie / u.mole
            self.MASS = u.grams / u.mole
            self.CHARGE = u.elementary_charge
        elif unit_set == 'metal':
            self.DIST = u.angstroms
            self.VEL = u.angstroms / u.picosecond
            self.ENERGY = u.joule / u.coulomb * u.elementary_charge
            self.MASS = u.grams / u.mole
            self.CHARGE = u.elementary_charge
        elif unit_set == 'si':
            self.DIST = u.meters
            self.VEL = u.meters / u.second
            self.ENERGY = u.joules
            self.MASS = u.kilograms
            self.CHARGE = u.coulomb
        elif unit_set == 'cgs':
            self.DIST = u.centimeter
            self.VEL = u.centimeter / u.second
            self.ENERGY = u.erg
            self.MASS = u.grams
            self.CHARGE = np.sqrt(u.erg * u.centimeter)
        elif unit_set == 'micro':
            self.DIST = u.micrometers
            self.VEL = u.nanometers / u.nanosecond
            self.ENERGY = u.picogram * (
                u.micrometer / u.microsecond) ^ 2
            self.MASS = u.attograms
            self.CHARGE = u.elementary_charge
        elif unit_set == 'nano':
            self.DIST = u.nanometers
            self.VEL = u.nanometer / u.nanosecond
            self.ENERGY = u.attogram * (
                u.nanometer / u.nanosecond) ^ 2
            self.MASS = u.attograms
            self.CHARGE = u.elementary_charge
        elif unit_set == 'lj':
            self.DIST = u.dimensionless
            self.VEL = u.dimensionless
            self.ENERGY = u.dimensionless
            self.MASS = u.dimensionless
            self.CHARGE = u.dimensionless
            warnings.warn("Using unit type lj: All values are dimensionless. "
                          "This is untested and will likely fail. "
                          "See LAMMPS doc for more.", LammpsWarning)
        elif unit_set == 'electron':
            self.DIST = u.bohr
            self.VEL = u.bohr / u.atu
            self.ENERGY = u.hartree
            self.MASS = u.amu
            self.CHARGE = u.elementary_charge
        else:
            raise LammpsError('Unsupported unit set specified: {0}'.format(unit_set))

    #===================================================

    def __init__(self, fname=None, parametrize=True):
        from parmed import load_file
        super(LammpsDataFile, self).__init__()
        self.parameterset = None
        if fname is not None:
            self.read(fname, parametrize)
            self.unchange()

    #===================================================

    def read(self, fname, parametrize=True):
        """ Reads the data file into the current instance """
        from parmed import lammps as lmp
        self.params = self.parameterset = ParameterSet()
        # bond_types = dict()
        # angle_types = dict()
        # ub_types = dict()
        # dihedral_types = dict()
        # exc_types = dict()
        # proper_multiterm_dihedrals = dict()
        self.masses = dict()
        self.box = [None] * 6

        # TODO: support other unit sets.
        # http://lammps.sandia.gov/doc/units.html
        self.set_units('real')

        self.parsable_keywords = {'Masses': self._parse_masses,
                                  'Pair Coeffs': self._parse_pair_coeffs,
                                  'Bond Coeffs': self._parse_bond_coeffs,
                                  'Angle Coeffs': self._parse_angle_coeffs,
                                  'Dihedral Coeffs': self._parse_dihedral_coeffs,
                                  'Improper Coeffs': self._parse_improper_coeffs}
        with open(fname) as f:
            self.title = next(f).strip()
            for line in f:
                line = line.partition('#')[0].strip()  # Remove trailing comments.
                if not line:
                    continue
                # Catch all box dimensions.
                if 'xlo' in line and 'xhi' in line:
                    self._parse_box(line.split(), 0)
                elif 'ylo' in line and 'yhi' in line:
                    self._parse_box(line.split(), 1)
                elif 'zlo' in line and 'zhi' in line:
                    self._parse_box(line.split(), 2)
                # Other headers.
                else:
                    keyword = line.partition('#')[0].strip()
                    if keyword in self.parsable_keywords:
                        self._parse_section(f, keyword)

        self.parsable_keywords = {'Atoms': self._parse_atoms,
                                  'Velocities': self._parse_velocities,
                                  'Bonds': self._parse_bonds,
                                  'Angles': self._parse_angles,
                                  'Dihedrals': self._parse_dihedrals,
                                  'Impropers': self._parse_impropers}
        with open(fname) as f:
            for line in f:
                keyword = line.partition('#')[0].strip()
                if keyword in self.parsable_keywords:
                    self._parse_section(f, keyword)

        if parametrize:
            self.parametrize()


    #===================================================

    # Private parsing helper functions

    def _parse_section(self, f, keyword):
        line = next(f).strip()  # Toss out blank line.
        if line:
            raise LammpsError('Expected blank line after {} keyword.'.format(keyword))

        for line in f:
            line = line.partition('#')[0]
            if not line.strip():
                break
            self.parsable_keywords[keyword](line)

    def _parse_box(self, line, dim):
        """Read a box line from the data file. """
        fields = [float(field) for field in line[:2]]
        box_length = fields[1] - fields[0]
        if box_length > 0:
            self.box[dim] = box_length * self.DIST.conversion_factor_to(u.nanometers)
        else:
            raise LammpsError("Negative box length specified in data file.")

    def _parse_masses(self, line):
        """Read a mass entry from the data file. """
        fields = line.split()
        self.masses[int(fields[0])] = float(fields[1]) * self.MASS

    def _parse_pair_coeffs(self, line):
        """Read pair coefficients from data file. """
        self.nb_types = dict()
        fields = [float(field) for field in line.partition('#')[0].split()]
        if len(self.pair_style) == 1:
            # TODO: lookup of type of pairstyle to determine format
            if self.system.nonbonded_function == 1:
                self.nb_types[int(fields[0])] = [fields[1] * self.ENERGY,
                                                 fields[2] * self.DIST]
            else:
                raise UnimplementedSetting(line, ENGINE)
        else:
            raise UnimplementedFunctional(line, ENGINE)

    def _parse_force_coeffs(self, data_lines, force_name, force_classes,
                           force_style, lammps_forces, canonical_force):
        """Read force coefficients from data file."""
        next(data_lines)  # toss out blank line

        for line in data_lines:
            if not line.strip():
                break  # found another blank line
            fields = line.partition('#')[0].split()

            warn = False
            if len(force_style) == 1:
                style = next(iter(force_style))  # There's only one entry.
                if style == fields[1]:
                    field_offset = 2
                else:
                    if re.search('[a-zA-Z]+', fields[1]):
                        if style == 'none':
                            style = fields[1]
                            field_offset = 2
                        else:
                            warn = True
                    else:
                        field_offset = 1

            elif len(force_style) > 1:
                style = fields[1]
                field_offset = 2
                if style not in force_style:
                    warn = True
            else:
                raise LammpsError("No entries found in '%s_style'." % (force_name))

            if warn:
                logger.warning('{0} type found in {1} Coeffs that was not '
                               'specified in {2}_style: {3}'.format(force_name, force_name, force_name, style))

            # what internal force correspond to this style
            force_class = lammps_forces[style]

            # Get the parameters from the line and translate into keywords
            kwds = self.create_kwds_from_entries(fields, force_class,
                                                 offset=field_offset)

            # translate the force into canonical form
            force_class, kwds = canonical_force(kwds, force_class,
                                                direction='into')
            # add to the dictionary of this force term
            force_classes[int(fields[0])] = [force_class, kwds]

    def _parse_bond_coeffs(self, line):
        pass

    def _parse_angle_coeffs(self, line):
        pass

    def _parse_dihedral_coeffs(self, line):
        pass

    def _parse_improper_coeffs(self, line):
        pass

    def _parse_atoms(self, line):
        """Read atoms from data file."""
        fields = line.partition('#')[0].split()

        # TODO: support other atom styles.
        # http://lammps.sandia.gov/doc/atom_style.html
        atom_type_int = int(fields[2])
        atom_type = 'lmp_{:03d}'.format(atom_type_int)
        mass = self.masses[atom_type_int]
        chg = float(fields[3])
        xx = float(fields[4]) * self.DIST
        xy = float(fields[5]) * self.DIST
        xz = float(fields[6]) * self.DIST

        atom = Atom(atomic_number=-1, name=atom_type, type=atom_type_int,
                    charge=chg, mass=mass)
        atom.xx = xx
        atom.xy = xy
        atom.xz = xz
        self.add_atom(atom, resname='RES', resnum=1)

    def _parse_velocities(self, line):
        """ """
        pass

    def _parse_bonds(self, line):
        """ """
        fields = [int(field) for field in line.partition('#')[0].split()]

        bond_type_num = int(fields[1])
        i, j = int(fields[2]), int(fields[3])
        atom_i, atom_j = self.atoms[i], self.atoms[j]
        bond = Bond(atom_i, atom_j)
        bond_type = None
        if len(fields) > 4:
            # create BondType
            pass
        self.bonds.append(bond)
        self.bond_types.append(bond_type)

    def _parse_angles(self, line):
        pass

    def _parse_dihedrals(self, line):
        pass

    def _parse_impropers(self, line):
        pass

    #===================================================

    def parametrize(self):
        """
        Assign parameters to the current structure. This should be called
        *after* `read`
        """
        if self.parameterset is None:
            raise RuntimeError('parametrize called before read')
        params = copy.copy(self.parameterset)
        def update_typelist_from(ptypes, types):
            added_types = set(id(typ) for typ in types)
            for k, typ in iteritems(ptypes):
                if not typ.used: continue
                if id(typ) in added_types: continue
                added_types.add(id(typ))
                types.append(typ)
            types.claim()
        # Assign all of the parameters. If they've already been assigned (i.e.,
        # on the parameter line itself) keep the existing parameters
        for atom in self.atoms:
            atom.atom_type = params.atom_types[atom.type]
        # The list of ordered 2-tuples of atoms explicitly specified in [ pairs ].
        # Under most circumstances, this is the list of 1-4 pairs.
        gmx_pair = set()
        for pair in self.adjusts:
            if pair.atom1 > pair.atom2:
                gmx_pair.add((pair.atom2, pair.atom1))
            else:
                gmx_pair.add((pair.atom1, pair.atom2))
            if pair.type is not None: continue
            key = (_gettype(pair.atom1), _gettype(pair.atom2))
            if key in params.pair_types:
                pair.type = params.pair_types[key]
                pair.type.used = True
            elif self.defaults.gen_pairs == 'yes':
                assert self.combining_rule in ('geometric', 'lorentz'), \
                        'Unrecognized combining rule'
                if self.combining_rule == 'geometric':
                    eps = math.sqrt(pair.atom1.epsilon * pair.atom2.epsilon)
                    sig = math.sqrt(pair.atom1.sigma * pair.atom2.sigma)
                elif self.combining_rule == 'lorentz':
                    eps = math.sqrt(pair.atom1.epsilon * pair.atom2.epsilon)
                    sig = 0.5 * (pair.atom1.sigma + pair.atom2.sigma)
                eps *= self.defaults.fudgeLJ
                pairtype = NonbondedExceptionType(sig*2**(1/6), eps,
                            self.defaults.fudgeQQ, list=self.adjust_types)
                self.adjust_types.append(pairtype)
                pair.type = pairtype
                pair.type.used = True
            else:
                raise ParameterError('Not all pair parameters can be found')
        update_typelist_from(params.pair_types, self.adjust_types)
        # This is the list of 1-4 pairs determined from the bond graph.
        # If this is different from what's in [ pairs ], we print a warning
        # and make some adjustments (specifically, other programs assume
        # the 1-4 list is complete, so we zero out the parameters for
        # 1-4 pairs that aren't in [ pairs ].
        true_14 = set()
        for bond in self.bonds:
            for bpi in bond.atom1.bond_partners:
                for bpj in bond.atom2.bond_partners:
                    if len(set([bpi, bond.atom1, bond.atom2, bpj])) < 4:
                        continue
                    if bpi in bpj.bond_partners or bpi in bpj.angle_partners:
                        continue
                    if bpi > bpj:
                        true_14.add((bpj, bpi))
                    else:
                        true_14.add((bpi, bpj))
            if bond.type is not None: continue
            key = (_gettype(bond.atom1), _gettype(bond.atom2))
            if key in params.bond_types:
                bond.type = params.bond_types[key]
                bond.type.used = True
            else:
                raise ParameterError('Not all bond parameters found')
        if len(true_14 - gmx_pair) > 0:
            zero_pairtype = NonbondedExceptionType(0.0, 0.0, 0.0,
                                                   list=self.adjust_types)
            self.adjust_types.append(zero_pairtype)
            num_zero_14 = 0
            for a1, a2 in (true_14 - gmx_pair):
                self.adjusts.append(NonbondedException(a1, a2, zero_pairtype))
                num_zero_14 += 1
            warnings.warn('%i 1-4 pairs were missing from the [ pairs ] '
                          'section and were set to zero; make sure you '
                          'know what you\'re doing!' % num_zero_14,
                          LammpsWarning)
        if len(gmx_pair - true_14) > 0:
            warnings.warn('The [ pairs ] section contains %i exceptions that '
                          'aren\'t 1-4 pairs; make sure you know what '
                          'you\'re doing!' % (len(gmx_pair - true_14)),
                          LammpsWarning)
        update_typelist_from(params.bond_types, self.bond_types)
        for angle in self.angles:
            if angle.type is not None: continue
            key = (_gettype(angle.atom1), _gettype(angle.atom2),
                   _gettype(angle.atom3))
            if key in params.angle_types:
                angle.type = params.angle_types[key]
                angle.type.used = True
            else:
                raise ParameterError('Not all angle parameters found')
        update_typelist_from(params.angle_types, self.angle_types)
        for ub in self.urey_bradleys:
            if ub.type is not None: continue
            key = (_gettype(ub.atom1), _gettype(ub.atom2))
            if key in params.urey_bradley_types:
                ub.type = params.urey_bradley_types[key]
                if ub.type is not NoUreyBradley:
                    ub.type.used = True
            else:
                raise ParameterError('Not all urey-bradley parameters found')
        # Now strip out all of the Urey-Bradley terms whose parameters are 0
        for i in reversed(range(len(self.urey_bradleys))):
            if self.urey_bradleys[i].type is NoUreyBradley:
                del self.urey_bradleys[i]
        update_typelist_from(params.urey_bradley_types, self.urey_bradley_types)
        for t in self.dihedrals:
            if t.type is not None: continue
            key = (_gettype(t.atom1), _gettype(t.atom2), _gettype(t.atom3),
                   _gettype(t.atom4))
            if not t.improper:
                wckey = ('X', _gettype(t.atom2), _gettype(t.atom3), 'X')
                wckey1 = (_gettype(t.atom1), _gettype(t.atom2),
                          _gettype(t.atom3), 'X')
                wckey2 = ('X', _gettype(t.atom2), _gettype(t.atom3),
                          _gettype(t.atom4))
                if key in params.dihedral_types:
                    t.type = params.dihedral_types[key]
                    t.type.used = True
                elif wckey1 in params.dihedral_types:
                    t.type = params.dihedral_types[wckey1]
                    t.type.used = True
                elif wckey2 in params.dihedral_types:
                    t.type = params.dihedral_types[wckey2]
                    t.type.used = True
                elif wckey in params.dihedral_types:
                    t.type = params.dihedral_types[wckey]
                    t.type.used = True
                else:
                    raise ParameterError('Not all torsion parameters found')
            else:
                if key in params.improper_periodic_types:
                    t.type = params.improper_periodic_types[key]
                    t.type.used = True
                else:
                    for wckey in [(key[0],key[1],key[2],'X'),
                                  ('X',key[1],key[2],key[3]),
                                  (key[0],key[1],'X','X'),
                                  ('X','X',key[2],key[3])]:
                        if wckey in params.improper_periodic_types:
                            t.type = params.improper_periodic_types[wckey]
                            t.type.used = True
                            break
                    else:
                        raise ParameterError('Not all improper torsion '
                                             'parameters found')
        update_typelist_from(params.dihedral_types, self.dihedral_types)
        update_typelist_from(params.improper_periodic_types, self.dihedral_types)
        for t in self.rb_torsions:
            if t.type is not None: continue
            key = (_gettype(t.atom1), _gettype(t.atom2), _gettype(t.atom3),
                   _gettype(t.atom4))
            wckey = ('X', _gettype(t.atom2), _gettype(t.atom3), 'X')
            wckey1 = (_gettype(t.atom1), _gettype(t.atom2),
                      _gettype(t.atom3), 'X')
            wckey2 = ('X', _gettype(t.atom2), _gettype(t.atom3),
                      _gettype(t.atom4))
            if key in params.rb_torsion_types:
                t.type = params.rb_torsion_types[key]
                t.type.used = True
            elif wckey1 in params.rb_torsion_types:
                t.type = params.rb_torsion_types[wckey1]
                t.type.used = True
            elif wckey2 in params.rb_torsion_types:
                t.type = params.rb_torsion_types[wckey2]
                t.type.used = True
            elif wckey in params.rb_torsion_types:
                t.type = params.rb_torsion_types[wckey]
                t.type.used = True
            else:
                raise ParameterError('Not all R-B torsion parameters found')
        update_typelist_from(params.rb_torsion_types, self.rb_torsion_types)
        self.update_dihedral_exclusions()
        for t in self.impropers:
            if t.type is not None: continue
            key = tuple(sorted([_gettype(t.atom1), _gettype(t.atom2),
                                _gettype(t.atom3), _gettype(t.atom4)]))
            if key in params.improper_types:
                t.type = params.improper_types[key]
                t.type.used = True
                continue
            # Now we will try to find a compatible wild-card... the first atom
            # is the central atom. So take each of the other three and plug that
            # one in
            for anchor in (_gettype(t.atom2), _gettype(t.atom3),
                           _gettype(t.atom4)):
                wckey = tuple(sorted([_gettype(t.atom1), anchor, 'X', 'X']))
                if wckey not in params.improper_types: continue
                t.type = params.improper_types[wckey]
                t.type.used = True
                break
            else:
                raise ParameterError('Not all improper parameters found')
        update_typelist_from(params.improper_types, self.improper_types)
        for c in self.cmaps:
            if c.type is not None: continue
            key = (_gettype(c.atom1), _gettype(c.atom2), _gettype(c.atom3),
                    _gettype(c.atom4), _gettype(c.atom5))
            if key in params.cmap_types:
                c.type = params.cmap_types[key]
                c.type.used = True
            else:
                raise ParameterError('Not all cmap parameters found')
        update_typelist_from(params.cmap_types, self.cmap_types)

    #===================================================

    def copy(self, cls, split_dihedrals=False):
        """
        Makes a copy of the current structure as an instance of a specified
        subclass

        Parameters
        ----------
        cls : Structure subclass
            The returned object is a copy of this structure as a `cls` instance
        split_dihedrals : ``bool``
            If True, then the Dihedral entries will be split up so that each one
            is paired with a single DihedralType (rather than a
            DihedralTypeList)

        Returns
        -------
        *cls* instance
            The instance of the Structure subclass `cls` with a copy of the
            current Structure's topology information
        """
        c = super(LammpsDataFile, self).copy(cls, split_dihedrals)
        c.defaults = copy.copy(self.defaults)
        return c

    #===================================================

    def __getitem__(self, selection):
        """ See Structure.__getitem__ for documentation """
        # Make sure defaults is properly copied
        struct = super(LammpsDataFile, self).__getitem__(selection)
        if isinstance(struct, Atom):
            return struct
        struct.defaults = copy.copy(self.defaults)
        return struct

    #===================================================

    @classmethod
    def from_structure(cls, struct, copy=False):
        """ Instantiates a LammpsDataFile instance from a Structure

        Parameters
        ----------
        struct : :class:`parmed.Structure`
            The input structure to generate from
        copy : bool, optional
            If True, assign from a *copy* of ``struct`` (this is a lot slower).
            Default is False

        Returns
        -------
        lmpdat : :class:`LammpsDataFile`
            The data file defined by the given struct
        """
        from copy import copy as _copy
        lmpdat = cls()
        if copy:
            struct = _copy(struct)
            struct.join_dihedrals()
        lmpdat.atoms = struct.atoms
        lmpdat.residues = struct.residues
        lmpdat.bonds = struct.bonds
        lmpdat.angles = struct.angles
        lmpdat.dihedrals = struct.dihedrals
        lmpdat.impropers = struct.impropers
        lmpdat.cmaps = struct.cmaps
        lmpdat.rb_torsions = struct.rb_torsions
        lmpdat.urey_bradleys = struct.urey_bradleys
        lmpdat.adjusts = struct.adjusts
        lmpdat.bond_types = struct.bond_types
        lmpdat.angle_types = struct.angle_types
        lmpdat.dihedral_types = struct.dihedral_types
        lmpdat.improper_types = struct.improper_types
        lmpdat.cmap_types = struct.cmap_types
        lmpdat.rb_torsion_types = struct.rb_torsion_types
        lmpdat.urey_bradley_types = struct.urey_bradley_types
        lmpdat.combining_rule = struct.combining_rule
        lmpdat.box = struct.box
        if (struct.trigonal_angles or
                struct.out_of_plane_bends or
                struct.pi_torsions or
                struct.stretch_bends or
                struct.torsion_torsions or
                struct.chiral_frames or
                struct.multipole_frames):
            raise TypeError('LammpsDataFile does not support Amoeba FF')
        lmpdat.parameterset = ParameterSet.from_structure(struct,
                                            allow_unequal_duplicates=True)
        return lmpdat

    #===================================================

    def write(self, dest, unit_set='real', parameters='inline'):
        """ Write a Lammps Data File from a Structure

        Parameters
        ----------
        dest : str or file-like
            The name of a file or a file object to write the Lammps data file to
        parameters : 'inline' or str or file-like object, optional
            This specifies where parameters should be printed. If 'inline'
            (default), the parameters are written on the same lines as the
            valence terms are defined on. Any other string is interpreted as a
            filename for an ITP that will be written to and then included at the
            top of `dest`. If it is a file-like object, parameters will be
            written there.  If parameters is the same as ``dest``, then the
            parameter types will be written to the same data file.

        """
        import parmed.lammps as lmp
        from parmed import __version__
        own_handle = False
        fname = ''
        params = ParameterSet.from_structure(self, allow_unequal_duplicates=True)
        if isinstance(dest, string_types):
            fname = '%s ' % dest
            dest = genopen(dest, 'w')
            own_handle = True
        elif not hasattr(dest, 'write'):
            raise TypeError('dest must be a file name or file-like object')

        # Determine where to write the parameters
        own_parfile_handle = False
        include_parfile = None
        if parameters == 'inline':
            parfile = dest
        elif isinstance(parameters, string_types):
            if parameters == fname.strip():
                parfile = dest
            else:
                own_parfile_handle = True
                parfile = genopen(parameters, 'w')
                include_parfile = parameters
        elif hasattr(parameters, 'write'):
            parfile = parameters
        else:
            raise ValueError('parameters must be "inline", a file name, or '
                             'a file-like object')

        self.set_units(unit_set)
        try:
            # Write the header
            now = datetime.now()
            dest.write("{} - created by ParmEd VERSION {} on {}.\n".format(
                fname, __version__, now.strftime('%a. %B  %w %X %Y')))

            # # Print all atom types
            # parfile.write('[ atomtypes ]\n')
            # if any(typ._bond_type is not None
            #         for key, typ in iteritems(params.atom_types)):
            #     print_bond_types = True
            # else:
            #     print_bond_types = False
            # if all(typ.atomic_number != -1
            #         for key, typ in iteritems(params.atom_types)):
            #     print_atnum = True
            # else:
            #     print_atnum = False
            # parfile.write('; name    ')
            # if print_bond_types:
            #     parfile.write('bond_type ')
            # if print_atnum:
            #     parfile.write('at.num    ')
            # parfile.write('mass    charge ptype  sigma      epsilon\n')
            # econv = u.kilocalories.conversion_factor_to(u.kilojoules)
            # for key, atom_type in iteritems(params.atom_types):
            #     parfile.write('%-7s ' % atom_type)
            #     if print_bond_types:
            #         parfile.write('%-8s ' % atom_type.bond_type)
            #     if print_atnum:
            #         parfile.write('%8d ' % atom_type.atomic_number)
            #     parfile.write('%10.5f  %10.6f  A %13.6g %13.6g\n' % (
            #                   atom_type.mass, atom_type.charge, atom_type.sigma/10,
            #                   atom_type.epsilon*econv))
            # parfile.write('\n')
            # # Print all parameter types unless we asked for inline
            # if parameters != 'inline':
            #     if params.bond_types:
            #         parfile.write('[ bondtypes ]\n')
            #         parfile.write('; i    j  func       b0          kb\n')
            #         used_keys = set()
            #         conv = (u.kilocalorie/u.angstrom**2).conversion_factor_to(
            #                     u.kilojoule/u.nanometer**2) * 2
            #         for key, param in iteritems(params.bond_types):
            #             if key in used_keys: continue
            #             used_keys.add(key)
            #             used_keys.add(tuple(reversed(key)))
            #             parfile.write('%-5s %-5s    1   %.5f   %f\n' % (key[0],
            #                           key[1], param.req/10, param.k*conv))
            #         parfile.write('\n')
            #     if params.pair_types and self.defaults.gen_pairs == 'no':
            #         parfile.write('[ pairtypes ]\n')
            #         parfile.write('; i j   func    sigma1-4    epsilon1-4 ;'
            #                       ' ; THESE ARE 1-4 INTERACTIONS\n')
            #         econv = u.kilocalorie.conversion_factor_to(u.kilojoule)
            #         lconv = u.angstrom.conversion_factor_to(u.nanometer)
            #         used_keys = set()
            #         for key, param in iteritems(params.pair_types):
            #             if key in used_keys: continue
            #             used_keys.add(key)
            #             used_keys.add(tuple(reversed(key)))
            #             parfile.write('%-5s %-5s  1  %.5f    %.5f\n' %
            #                           (key[0], key[1], param.sigma*lconv,
            #                            param.epsilon*econv))
            #         parfile.write('\n')
            #     if params.angle_types:
            #         parfile.write('[ angletypes ]\n')
            #         parfile.write(';  i    j    k  func       th0       cth '
            #                       '   rub         kub\n')
            #         used_keys = set()
            #         conv = (u.kilocalorie/u.radian**2).conversion_factor_to(
            #                     u.kilojoule/u.radian**2) * 2
            #         bconv = (u.kilocalorie/u.angstrom**2).conversion_factor_to(
            #                     u.kilojoule/u.nanometer**2) * 2
            #         for key, param in iteritems(params.angle_types):
            #             if key in used_keys: continue
            #             used_keys.add(key)
            #             used_keys.add(tuple(reversed(key)))
            #             part = '%-5s %-5s %-5s    %%d   %8.3f   %8.3f' % (
            #                     key[0], key[1], key[2], param.theteq,
            #                     param.k*conv)
            #             if (key[0], key[2]) in params.urey_bradley_types:
            #                 ub = params.urey_bradley_types[(key[0], key[2])]
            #                 parfile.write(part % 5)
            #                 parfile.write('  %8.3f  %8.3f\n' % (ub.req/10,
            #                               ub.k*bconv))
            #             else:
            #                 parfile.write(part % 1)
            #                 parfile.write('\n')
            #         parfile.write('\n')
            #     if params.dihedral_types:
            #         parfile.write('[ dihedraltypes ]\n')
            #         parfile.write(';i  j   k  l  func      phase      kd      '
            #                       'pn\n')
            #         used_keys = set()
            #         conv = u.kilocalories.conversion_factor_to(u.kilojoules)
            #         fmt = '%-6s %-6s %-6s %-6s  %d   %.2f   %.6f   %d\n'
            #         for key, param in iteritems(params.dihedral_types):
            #             if key in used_keys: continue
            #             used_keys.add(key)
            #             used_keys.add(tuple(reversed(key)))
            #             for dt in param:
            #                 parfile.write(fmt % (key[0], key[1], key[2],
            #                               key[3], 9, dt.phase,
            #                               dt.phi_k*conv, int(dt.per)))
            #         parfile.write('\n')
            #     if params.improper_periodic_types:
            #         parfile.write('[ dihedraltypes ]\n')
            #         parfile.write(';i  j   k  l  func      phase      kd      '
            #                       'pn\n')
            #         used_keys = set()
            #         conv = u.kilojoules.conversion_factor_to(u.kilocalories)
            #         fmt = '%-6s %-6s %-6s %-6s  %d   %.2f   %.6f   %d\n'
            #         for key, param in iteritems(params.improper_periodic_types):
            #             if key in used_keys: continue
            #             used_keys.add(key)
            #             used_keys.add(tuple(reversed(key)))
            #             parfile.write(fmt % (key[0], key[1], key[2], key[3],
            #                           4, param.phase, param.phi_k*conv,
            #                           int(param.per)))
            #         parfile.write('\n')
            #     if params.improper_types:
            #         # BUGBUG -- The ordering is borked here because that made it
            #         # simpler for me to work with back when I wrote the CHARMM
            #         # parsers. This needs to be fixed now and handled correctly.
            #         parfile.write('[ dihedraltypes ]\n')
            #         parfile.write('; i  j       k       l       func     q0    '
            #                       'cq\n')
            #         fmt = '%-6s %-6s %-6s %-6s    %d   %.4f   %.4f\n'
            #         conv = u.kilocalories.conversion_factor_to(u.kilojoules)*2
            #         for key, param in iteritems(params.improper_types):
            #             parfile.write(fmt % (key[0], key[1], key[2], key[3],
            #                           2, param.psi_eq, param.psi_k*conv))
            #         parfile.write('\n')
            # # CMAP grids are never printed inline, so if we have them, we need
            # # to write a dedicated section for them
            # if params.cmap_types:
            #         parfile.write('[ cmaptypes ]\n\n')
            #         used_keys = set()
            #         conv = u.kilocalories.conversion_factor_to(u.kilojoules)
            #         for key, param in iteritems(params.cmap_types):
            #             if key in used_keys: continue
            #             used_keys.add(key)
            #             used_keys.add(tuple(reversed(key)))
            #             parfile.write('%-6s %-6s %-6s %-6s %-6s   1   '
            #                           '%4d %4d' % (key[0], key[1], key[2],
            #                           key[3], key[4], param.resolution,
            #                           param.resolution))
            #             res2 = param.resolution * param.resolution
            #             for i in range(0, res2, 10):
            #                 parfile.write('\\\n')
            #                 end = min(i+10, res2)
            #                 parfile.write(' '.join(str(param.grid[j]*conv)
            #                               for j in range(i, end)))
            #             parfile.write('\n\n')

            dest.write('\nAtoms\n\n')
            runchg = 0
            for residue in self.residues:
                for atom in residue:
                    runchg += atom.charge
                    dest.write('{0:-6d} {1:-6d} {2:-6d} {3:5.8f} {4:12.7f}'
                               ' {5:12.7f} {6:12.7f}  # qtot {7:.4f}\n'.format(
                        atom.idx + 1,
                        residue.idx + 1,
                        #atom.type.n
                        1,
                        atom.charge,
                        atom.xx.value_in_unit(self.DIST),
                        atom.xy.value_in_unit(self.DIST),
                        atom.xz.value_in_unit(self.DIST),
                        runchg))
            print('total charge: ', runchg)
            if self.bonds:
                dest.write('\nBonds\n\n')
                for n, bond in enumerate(self.bonds):
                    dest.write('{:d} {:d} {:d} {:d}\n'.format(
                        n,
                        #bond.type.n,
                        1,
                        bond.atom1.idx+1,
                        bond.atom2.idx+1))
            print('n_bonds: ', n)
            # # Angles
            # if struct.angles:
            #     conv = (u.kilocalorie_per_mole/u.radian**2).conversion_factor_to(
            #                 u.kilojoule_per_mole/u.radian**2)*2
            #     conv2 = (u.kilocalorie_per_mole/u.angstrom**2).conversion_factor_to(
            #             u.kilojoule_per_mole/u.nanometer**2)*2
            #     dest.write('[ angles ]\n')
            #     dest.write(';%6s %6s %6s %5s %10s %10s %10s %10s\n' %
            #                ('ai', 'aj', 'ak', 'funct', 'c0', 'c1', 'c2', 'c3'))
            #     for angle in struct.angles:
            #         dest.write('%7d %6d %6d %5d' % (angle.atom1.idx+1,
            #                    angle.atom2.idx+1, angle.atom3.idx+1,
            #                    angle.funct))
            #         if angle.type is None:
            #             dest.write('\n')
            #             continue
            #         key = (_gettype(angle.atom1), _gettype(angle.atom2),
            #                _gettype(angle.atom3))
            #         param_equal = (key in params.angle_types and
            #                             params.angle_types[key] == angle.type)
            #         if angle.funct == 5:
            #             # Find the Urey-Bradley term, if it exists
            #             for ub in struct.urey_bradleys:
            #                 if angle.atom1 in ub and angle.atom3 in ub:
            #                     ubtype = ub.type
            #                     break
            #             else:
            #                 ubtype = NoUreyBradley
            #             ubkey = (key[0], key[2])
            #             param_equal = param_equal and (
            #                     ubkey in params.urey_bradley_types and
            #                     ubtype == params.urey_bradley_types[ubkey])
            #         if writeparams or not param_equal:
            #             dest.write('   %.5f %f' % (angle.type.theteq,
            #                                        angle.type.k*conv))
            #             if angle.funct == 5:
            #                 dest.write(' %.5f %f' % (ubtype.req/10, ubtype.k*conv2))
            #         dest.write('\n')
            #     dest.write('\n')
            # # Dihedrals
            # if struct.dihedrals:
            #     dest.write('[ dihedrals ]\n')
            #     dest.write((';%6s %6s %6s %6s %5s'+' %10s'*6) % ('ai', 'aj',
            #                'ak', 'al', 'funct', 'c0', 'c1', 'c2', 'c3',
            #                'c4', 'c5'))
            #     dest.write('\n')
            #     conv = u.kilocalories.conversion_factor_to(u.kilojoules)
            #     for dihed in struct.dihedrals:
            #         dest.write('%7d %6d %6d %6d %5d' % (dihed.atom1.idx+1,
            #                    dihed.atom2.idx+1, dihed.atom3.idx+1,
            #                    dihed.atom4.idx+1, dihed.funct))
            #         if dihed.type is None:
            #             dest.write('\n')
            #             continue
            #         if dihed.improper:
            #             typedict = params.improper_periodic_types
            #         else:
            #             typedict = params.dihedral_types
            #         key = (_gettype(dihed.atom1), _gettype(dihed.atom2),
            #                 _gettype(dihed.atom3), _gettype(dihed.atom4))
            #         if writeparams or key not in typedict or \
            #                 _diff_diheds(dihed.type, typedict[key]):
            #             if isinstance(dihed.type, DihedralTypeList):
            #                 dest.write('  %.5f  %.5f  %d' % (dihed.type[0].phase,
            #                     dihed.type[0].phi_k*conv, int(dihed.type[0].per)))
            #                 for dt in dihed.type[1:]:
            #                     dest.write('\n%7d %6d %6d %6d %5d  %.5f  %.5f  %d' %
            #                             (dihed.atom1.idx+1, dihed.atom2.idx+1,
            #                              dihed.atom3.idx+1, dihed.atom4.idx+1,
            #                              dihed.funct, dt.phase, dt.phi_k*conv,
            #                              int(dt.per)))
            #             else:
            #                 dest.write('  %.5f  %.5f  %d' % (dihed.type.phase,
            #                     dihed.type.phi_k*conv, int(dihed.type.per)))
            #         dest.write('\n')
            #     dest.write('\n')
            # # RB-torsions
            # if struct.rb_torsions:
            #     dest.write('[ dihedrals ]\n')
            #     dest.write((';%6s %6s %6s %6s %5s'+' %10s'*6) % ('ai', 'aj',
            #                'ak', 'al', 'funct', 'c0', 'c1', 'c2', 'c3',
            #                'c4', 'c5'))
            #     dest.write('\n')
            #     conv = u.kilocalories.conversion_factor_to(u.kilojoules)
            #     paramfmt = '  %12.5f  %12.5f  %12.5f  %12.5f  %12.5f  %12.5f'
            #     for dihed in struct.rb_torsions:
            #         dest.write('%7d %6d %6d %6d %5d' % (dihed.atom1.idx+1,
            #                    dihed.atom2.idx+1, dihed.atom3.idx+1,
            #                    dihed.atom4.idx+1, dihed.funct))
            #         if dihed.type is None:
            #             dest.write('\n')
            #             continue
            #         key = (_gettype(dihed.atom1), _gettype(dihed.atom2),
            #                 _gettype(dihed.atom3), _gettype(dihed.atom4))
            #         if writeparams or key not in params.rb_torsion_types or \
            #                 params.rb_torsion_types[key] != dihed.type:
            #             dest.write(paramfmt % (dihed.type.c0*conv,
            #                                    dihed.type.c1*conv,
            #                                    dihed.type.c2*conv,
            #                                    dihed.type.c3*conv,
            #                                    dihed.type.c4*conv,
            #                                    dihed.type.c5*conv))
            #             dest.write('\n')
            #     dest.write('\n')

        finally:
            if own_handle:
                dest.close()
            if own_parfile_handle:
                parfile.close()
    #===================================================

    def __getstate__(self):
        d = Structure.__getstate__(self)
        d['parameterset'] = self.parameterset
        d['defaults'] = self.defaults
        return d

    def __setstate__(self, d):
        Structure.__setstate__(self, d)
        self.parameterset = d['parameterset']
        self.defaults = d['defaults']

def _any_atoms_farther_than(structure, limit=3):
    """
    This function checks to see if there are any atom pairs farther away in the
    bond graph than the desired limit

    Parameters
    ----------
    structure : :class:`Structure`
        The structure to search through
    limit : int, optional
        The most number of bonds away to check for. Default is 3

    Returns
    -------
    within : bool
        True if any atoms are *more* than ``limit`` bonds away from any other
        atom
    """
    import sys
    if len(structure.atoms) <= limit + 1: return False
    sys.setrecursionlimit(max(sys.getrecursionlimit(), limit+1))
    for atom in structure.atoms:
        for atom in structure.atoms: atom.marked = limit + 1
        _mark_graph(atom, 0)
        if any((atom.marked > limit for atom in structure.atoms)):
            return True
    return False

def _mark_graph(atom, num):
    """ Marks all atoms in the graph listing the minimum number of bonds each
    atom is away from the current atom

    Parameters
    ----------
    atom : :class:`Atom`
        The current atom to evaluate in the bond graph
    num : int
        The current depth in our search
    limit : int
        The maximum depth we want to search
    """
    atom.marked = num
    for a in atom.bond_partners:
        if a.marked <= num: continue
        _mark_graph(a, num+1)

def _diff_diheds(dt1, dt2):
    """ Determine if 2 dihedrals are *really* different. dt1 can either be a
    DihedralType or a DihedralTypeList or dt1 can be a DihedralType and dt2 can
    be a DihedralTypeList.  This returns True if dt1 == dt2 *or* dt1 is equal to
    the only element of dt2
    """
    if type(dt1) is type(dt2) and dt1 == dt2:
        return False
    if isinstance(dt2, DihedralTypeList) and isinstance(dt1, DihedralType):
        if len(dt2) == 1 and dt2[0] == dt1: return False
    return True

def _gettype(atom):
    if atom.atom_type not in (None, UnassignedAtomType):
        return atom.atom_type.bond_type
    return atom.type
