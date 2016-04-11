"""
Tests the functionality in the parmed.lammps package
"""
from contextlib import closing
import copy
import numpy as np
import os
from parmed import (load_file, Structure, ExtraPoint, DihedralTypeList, Atom,
                    ParameterSet, Bond, NonbondedException, DihedralType,
                    RBTorsionType, Improper, Cmap, UreyBradley, BondType,
                    UnassignedAtomType, NonbondedExceptionType, NoUreyBradley)
from parmed.charmm import CharmmParameterSet
from parmed.exceptions import LammpsWarning, LammpsError, ParameterError
from parmed.lammps import LammpsDataFile, LammpsInputFile
from parmed import lammps as lmp
from parmed.topologyobjects import UnassignedAtomType
from parmed.utils.six.moves import range, zip, StringIO
import unittest
from utils import (get_fn, diff_files, get_saved_fn, FileIOTestCase, HAS_LAMMPS,
                   create_random_structure)
import utils
import warnings

@unittest.skipUnless(HAS_LAMMPS, "Cannot run LAMMPS tests without Lammps")
class TestLammpsData(FileIOTestCase):
    """ Tests the Lammps topology file parser """

    def setUp(self):
        warnings.filterwarnings('error', category=LammpsWarning)
        FileIOTestCase.setUp(self)

    def tearDown(self):
        warnings.filterwarnings('always', category=LammpsWarning)
        FileIOTestCase.tearDown(self)

    def _charmm27_checks(self, top):
        # Check that the number of terms are correct
        self.assertEqual(len(top.atoms), 1960)
        self.assertEqual(len(top.bonds), 1984)
        self.assertEqual(len(top.angles), 3547)
        self.assertEqual(len(top.dihedrals), 5187)
        self.assertEqual(len(top.impropers), 351)
        self.assertEqual(len(top.cmaps), 127)
        self.assertEqual(len(top.adjusts), 5106)
        self.assertFalse(top.unknown_functional)
        # Check the first and last of most of the terms to make sure that they
        # are the same as what is defined in the topology file
        self.assertEqual(top.atoms[0].type, 'NH3')
        self.assertEqual(top.atoms[0].name, 'N')
        self.assertEqual(top.atoms[0].mass, 14.007)
        self.assertEqual(top.atoms[0].charge, -0.3)
        self.assertEqual(top.atoms[0].atomic_number, 7)
        self.assertEqual(top.atoms[0].residue.name, 'LYS')
        self.assertEqual(top.atoms[1].type, 'HC')
        self.assertEqual(top.atoms[1].name, 'H1')
        self.assertEqual(top.atoms[1].mass, 1.008)
        self.assertEqual(top.atoms[1].charge, 0.33)
        self.assertEqual(top.atoms[1].atomic_number, 1)
        self.assertEqual(top.atoms[1].residue.name, 'LYS')
        self.assertEqual(top.atoms[1958].type, 'OC')
        self.assertEqual(top.atoms[1958].name, 'OT1')
        self.assertEqual(top.atoms[1958].mass, 15.9994)
        self.assertEqual(top.atoms[1958].charge, -0.67)
        self.assertEqual(top.atoms[1958].atomic_number, 8)
        self.assertEqual(top.atoms[1958].residue.name, 'LEU')
        self.assertEqual(top.atoms[1959].type, 'OC')
        self.assertEqual(top.atoms[1959].name, 'OT2')
        self.assertEqual(top.atoms[1959].mass, 15.9994)
        self.assertEqual(top.atoms[1959].charge, -0.67)
        self.assertEqual(top.atoms[1959].atomic_number, 8)
        self.assertEqual(top.atoms[1959].residue.name, 'LEU')
        # Bonds
        self.assertIs(top.bonds[0].atom1, top.atoms[0])
        self.assertIs(top.bonds[0].atom2, top.atoms[1])
        self.assertEqual(top.bonds[0].funct, 1)
        self.assertIs(top.bonds[1983].atom1, top.atoms[1957])
        self.assertIs(top.bonds[1983].atom2, top.atoms[1959])
        self.assertEqual(top.bonds[1983].funct, 1)
        # Angles
        self.assertIs(top.angles[0].atom1, top.atoms[1])
        self.assertIs(top.angles[0].atom2, top.atoms[0])
        self.assertIs(top.angles[0].atom3, top.atoms[2])
        self.assertEqual(top.angles[0].funct, 5)
        self.assertIs(top.angles[3546].atom1, top.atoms[1958])
        self.assertIs(top.angles[3546].atom2, top.atoms[1957])
        self.assertIs(top.angles[3546].atom3, top.atoms[1959])
        self.assertEqual(top.angles[0].funct, 5)
        # Dihedrals
        self.assertIs(top.dihedrals[0].atom1, top.atoms[1])
        self.assertIs(top.dihedrals[0].atom2, top.atoms[0])
        self.assertIs(top.dihedrals[0].atom3, top.atoms[4])
        self.assertIs(top.dihedrals[0].atom4, top.atoms[5])
        self.assertEqual(top.dihedrals[0].funct, 9)
        self.assertIs(top.dihedrals[5186].atom1, top.atoms[1949])
        self.assertIs(top.dihedrals[5186].atom2, top.atoms[1947])
        self.assertIs(top.dihedrals[5186].atom3, top.atoms[1953])
        self.assertIs(top.dihedrals[5186].atom4, top.atoms[1956])
        self.assertEqual(top.dihedrals[5186].funct, 9)
        # Impropers
        self.assertIs(top.impropers[0].atom1, top.atoms[22])
        self.assertIs(top.impropers[0].atom2, top.atoms[4])
        self.assertIs(top.impropers[0].atom3, top.atoms[24])
        self.assertIs(top.impropers[0].atom4, top.atoms[23])
        self.assertEqual(top.impropers[0].funct, 2)
        self.assertIs(top.impropers[350].atom1, top.atoms[1957])
        self.assertIs(top.impropers[350].atom2, top.atoms[1942])
        self.assertIs(top.impropers[350].atom3, top.atoms[1959])
        self.assertIs(top.impropers[350].atom4, top.atoms[1958])
        self.assertEqual(top.impropers[350].funct, 2)
        # Cmaps
        self.assertIs(top.cmaps[0].atom1, top.atoms[22])
        self.assertIs(top.cmaps[0].atom2, top.atoms[24])
        self.assertIs(top.cmaps[0].atom3, top.atoms[26])
        self.assertIs(top.cmaps[0].atom4, top.atoms[38])
        self.assertIs(top.cmaps[0].atom5, top.atoms[40])
        self.assertEqual(top.cmaps[0].funct, 1)
        self.assertIs(top.cmaps[126].atom1, top.atoms[1914])
        self.assertIs(top.cmaps[126].atom2, top.atoms[1916])
        self.assertIs(top.cmaps[126].atom3, top.atoms[1918])
        self.assertIs(top.cmaps[126].atom4, top.atoms[1938])
        self.assertIs(top.cmaps[126].atom5, top.atoms[1940])
        self.assertEqual(top.cmaps[126].funct, 1)
        # Adjusts
        self.assertIs(top.adjusts[0].atom1, top.atoms[0])
        self.assertIs(top.adjusts[0].atom2, top.atoms[7])
        self.assertEqual(top.adjusts[0].funct, 1)
        self.assertIs(top.adjusts[5105].atom1, top.atoms[1952])
        self.assertIs(top.adjusts[5105].atom2, top.atoms[1953])
        self.assertEqual(top.adjusts[5105].funct, 1)

    def test_charmm27_top(self):
        """ Tests parsing a Lammps topology with CHARMM 27 FF """
        top = LammpsDataFile(get_fn('1aki.charmm27.top'))
        self.assertEqual(top.combining_rule, 'lorentz')
        self.assertEqual(top.itps, ['charmm27.ff/forcefield.itp',
                                    'charmm27.ff/tip3p.itp',
                                    'charmm27.ff/ions.itp'])
        self._charmm27_checks(top)

    def test_lammps_data_detection(self):
        """ Tests automatic file detection of LAMMPS data files """
        fn = get_fn('test.top', written=True)
        with open(fn, 'w') as f:
            f.write('# not a gromacs topology file\n')
        self.assertFalse(LammpsDataFile.id_format(fn))
        with open(fn, 'w') as f:
            pass
        self.assertFalse(LammpsDataFile.id_format(fn))

    def test_write_charmm27_top(self):
        """ Tests writing a Lammps topology file with CHARMM 27 FF """
        top = load_file(get_fn('1aki.charmm27.top'))
        self.assertEqual(top.combining_rule, 'lorentz')
        LammpsDataFile.write(top,
                get_fn('1aki.charmm27.top', written=True))
        top2 = load_file(get_fn('1aki.charmm27.top', written=True))
        self._charmm27_checks(top)

    def _check_ff99sbildn(self, top):
        self.assertEqual(len(top.atoms), 4235)
        self.assertEqual(len(top.residues), 1046)
        self.assertEqual(sum(1 for a in top.atoms if isinstance(a, ExtraPoint)),
                         1042)
        self.assertEqual(len(top.bonds), 3192)
        self.assertEqual(len(top.angles), 1162)
        self.assertEqual(len(top.dihedrals), 179)

    def _check_equal_structures(self, top1, top2):
        def cmp_atoms(a1, a2):
            self.assertEqual(a1.name, a2.name)
            self.assertEqual(a1.mass, a2.mass)
            self.assertEqual(a1.atom_type, a2.atom_type)
            self.assertEqual(a1.type, a2.type)
            self.assertEqual(a1.charge, a2.charge)
            self.assertEqual(a1.atomic_number, a2.atomic_number)
            self.assertEqual(a1.residue.name, a2.residue.name)
            self.assertEqual(a1.residue.idx, a2.residue.idx)

        def cmp_valence(val1, val2, typeattrs=None):
            self.assertEqual(len(val1), len(val2))
            for v1, v2 in zip(val1, val2):
                self.assertIs(type(v1), type(v2))
                attrs = [attr for attr in dir(v1) if attr.startswith('atom')]
                atoms1 = [getattr(v1, attr) for attr in attrs]
                atoms2 = [getattr(v2, attr) for attr in attrs]
                for a1, a2 in zip(atoms1, atoms2):
                    cmp_atoms(a1, a2)
                # Check the type lists
                if typeattrs is not None:
                    for attr in typeattrs:
                        self.assertAlmostEqual(getattr(v1.type, attr),
                                               getattr(v2.type, attr), places=5)
                else:
                    self.assertEqual(v1.type, v2.type)

        def cmp_dihedrals(dih1, dih2):
            self.assertEqual(len(dih1), len(dih2))
            for v1, v2 in zip(dih1, dih2):
                self.assertIs(type(v1), type(v2))
                self.assertIs(type(v1.type), type(v2.type))
                atoms1 = [v1.atom1, v1.atom2, v1.atom3, v1.atom4]
                atoms2 = [v2.atom1, v2.atom2, v2.atom3, v2.atom4]
                for a1, a2 in zip(atoms1, atoms2):
                    cmp_atoms(a1, a2)
                self.assertEqual(v1.improper, v2.improper)
                self.assertEqual(v1.ignore_end, v2.ignore_end)
                if isinstance(v1, DihedralTypeList):
                    self.assertEqual(len(v1.type), len(v2.type))
                    for vt1, vt2 in zip(v1.type, v2.type):
                        self.assertAlmostEqual(v1.type.phi_k, v2.type.phi_k, places=5)
                        self.assertAlmostEqual(v1.type.per, v2.type.per, places=5)
                        self.assertAlmostEqual(v1.type.phase, v2.type.phase, places=5)

        self.assertEqual(len(top1.atoms), len(top2.atoms))
        for a1, a2 in zip(top1.atoms, top2.atoms):
            cmp_atoms(a1, a2)
        cmp_valence(top1.bonds, top2.bonds, ['k', 'req'])
        cmp_valence(top1.angles, top2.angles, ['k', 'theteq'])
        cmp_dihedrals(top1.dihedrals, top2.dihedrals)

    def test_read_amber99SBILDN(self):
        """ Tests parsing a Lammps topology with Amber99SBILDN and water """
        top = load_file(get_fn('ildn.solv.top'))
        self.assertEqual(top.combining_rule, 'lorentz')
        self._check_ff99sbildn(top)
        dts = top.dihedral_types[:]
        top.join_dihedrals()
        for dt1, dt2 in zip(dts, top.dihedral_types):
            self.assertIs(dt1, dt2)

    def test_write_amber99SBILDN(self):
        """ Tests writing a Lammps topology with multiple molecules """
        top = load_file(get_fn('ildn.solv.top'))
        self.assertEqual(top.combining_rule, 'lorentz')
        fn = get_fn('ildn.solv.top', written=True)
        top.write(fn, combine=None)
        top2 = load_file(fn)
        self._check_ff99sbildn(top2)
        self._check_equal_structures(top, top2)
        # Turn off gen_pairs and write another version
        top.defaults.gen_pairs = 'no'
        top.write(fn)
        top3 = load_file(fn)
        self._check_ff99sbildn(top3)
        self._check_equal_structures(top, top3)

    def test_duplicate_system_names(self):
        """ Tests that Lammps topologies never have duplicate moleculetypes """
        parm = load_file(get_fn('phenol.prmtop'))
        parm = parm * 20 + load_file(get_fn('biphenyl.prmtop')) * 20
        top = LammpsDataFile.from_structure(parm)
        self.assertEqual(top.combining_rule, 'lorentz')
        top.write(get_fn('phenol_biphenyl.top', written=True))
        top2 = LammpsDataFile(get_fn('phenol_biphenyl.top', written=True))
        self.assertEqual(len(top.residues), 40)

        # Now test this when we use "combine"
        warnings.filterwarnings('ignore', category=LammpsWarning)
        parm = load_file(os.path.join(get_fn('12.DPPC'), 'topol3.top'))
        fn = get_fn('samename.top', written=True)
        parm.residues[3].name = 'SOL' # Rename a DPPC to SOL
        parm.write(fn, combine=[[0, 1]])
        parm2 = load_file(fn)
        self.assertEqual(len(parm2.atoms), len(parm.atoms))
        self.assertEqual(len(parm2.residues), len(parm2.residues))
        for a1, a2 in zip(parm.atoms, parm2.atoms):
            self._equal_atoms(a1, a2)
        for r1, r2 in zip(parm.residues, parm2.residues):
            self.assertEqual(len(r1), len(r2))
            self.assertEqual(r1.name, r2.name)
            for a1, a2 in zip(r1, r2):
                self._equal_atoms(a1, a2)

    def test_lammps_data_from_structure(self):
        """ Tests the LammpsDataFile.from_structure constructor """
        struct = create_random_structure(True)
        self.assertRaises(TypeError, lambda:
                LammpsDataFile.from_structure(struct))
        parm = load_file(get_fn('ash.parm7'))
        parm.dihedrals[0].type.scee = 8.0
        self.assertRaises(LammpsError, lambda:
                LammpsDataFile.from_structure(parm))
        for dt in parm.dihedral_types: dt.scee = dt.scnb = 0
        top = LammpsDataFile.from_structure(parm)
        self.assertEqual(top.defaults.fudgeLJ, 1.0)
        self.assertEqual(top.defaults.fudgeQQ, 1.0)

    def test_OPLS(self):
        """ Tests the geometric combining rules in Lammps with OPLS/AA """
        parm = load_file(os.path.join(get_fn('05.OPLS'), 'topol.top'),
                         xyz=os.path.join(get_fn('05.OPLS'), 'conf.gro'))
        self.assertEqual(parm.combining_rule, 'geometric')
        self.assertEqual(parm.defaults.comb_rule, 3)
        parm.write(get_fn('test.topol', written=True), combine='all')
        parm2 = load_file(get_fn('test.topol', written=True))
        self.assertEqual(len(parm.atoms), len(parm2.atoms))
        # Check that the charge attribute is read correctly
        self.assertEqual(parm.parameterset.atom_types['opls_001'].charge, 0.5)
        # Check the xyz argument in the constructor being coordinates
        parm2 = load_file(os.path.join(get_fn('05.OPLS'), 'topol.top'),
                          xyz=parm.coordinates, box=[10, 10, 10, 90, 90, 90])
        np.testing.assert_equal(parm2.coordinates, parm.coordinates)
        np.testing.assert_equal(parm2.box, [10, 10, 10, 90, 90, 90])
        # Check the copy constructor
        p2 = LammpsDataFile.from_structure(parm, copy=True)
        self.assertEqual(p2.combining_rule, 'geometric')
        self.assertEqual(p2.defaults.comb_rule, 3)
        self.assertEqual(len(p2.atoms), len(parm.atoms))
        for a1, a2 in zip(p2.atoms, parm.atoms):
            self.assertIsNot(a1, a2)
            self.assertEqual(a1.name, a2.name)
            self.assertEqual(a1.type, a2.type)
            self.assertEqual(a1.atomic_number, a2.atomic_number)
            self.assertEqual(a1.mass, a2.mass)
        np.testing.assert_equal(p2.box, parm.box)

    def test_write_settles(self):
        """ Tests that settles is only written for water """
        fn = get_fn('blah.top', written=True)
        parm = load_file(os.path.join(get_fn('01.1water'), 'topol.top'))
        parm[0].atomic_number = parm[0].atom_type.atomic_number = 7
        parm.write(fn)
        with closing(LammpsDataFile(fn)) as f:
            for line in f:
                self.assertNotIn('settles', line)

    def test_write_extra_points(self):
        """ Test writing of LAMMPS files with virtual sites """
        f = StringIO('; TIP4Pew water molecule\n#include "amber99sb.ff/forcefield.itp"\n'
                     '#include "amber99sb.ff/tip4pew.itp"\n[ system ]\nWATER\n'
                     '[ molecules ]\nSOL 1\n')
        parm = LammpsDataFile(f)
        fn = get_fn('test.top', written=True)
        parm.write(fn)
        parm2 = load_file(fn)
        self.assertEqual(len(parm.atoms), len(parm2.atoms))
        self.assertEqual(len(parm.bonds), len(parm2.bonds))

    def test_without_parametrize(self):
        """ Tests loading a Lammps topology without parametrizing """
        parm = load_file(os.path.join(get_fn('05.OPLS'), 'topol.top'),
                         xyz=os.path.join(get_fn('05.OPLS'), 'conf.gro'),
                         parametrize=False)
        self.assertIs(parm.atoms[0].atom_type, UnassignedAtomType)
        self.assertTrue(all(x.type is None for x in parm.bonds))
        # Now try writing it out again
        fn = get_fn('test.top', written=True)
        parm.write(fn)

        parm = load_file(os.path.join(get_fn('04.Ala'), 'topol.top'),
                         parametrize=False)
        # Add an improper
        parm.impropers.append(Improper(*parm.atoms[:4]))
        parm.write(fn)

    def test_bad_data_loads(self):
        """ Test error catching in LammpsDataFile reading """
        fn = os.path.join(get_fn('03.AlaGlu'), 'topol.top')
        self.assertRaises(TypeError, lambda: load_file(fn, xyz=fn))
        self.assertRaises(ValueError, lambda: LammpsDataFile(xyz=1, box=1))
        wfn = os.path.join(get_fn('gmxtops'), 'duplicated_topol.top')
        self.assertRaises(LammpsError, lambda: LammpsDataFile(wfn))
        f = StringIO('\n[ defaults ]\n; not enough data\n 1\n\n')
        self.assertRaises(LammpsError, lambda: LammpsDataFile(f))
        f = StringIO('\n[ defaults ]\n; unsupported function type\n 2 1 yes\n')
        self.assertRaises(LammpsWarning, lambda: LammpsDataFile(f))
        warnings.filterwarnings('ignore', category=LammpsWarning)
        f.seek(0)
        self.assertTrue(LammpsDataFile(f).unknown_functional)
        warnings.filterwarnings('error', category=LammpsWarning)
        fn = os.path.join(get_fn('gmxtops'), 'bad_vsites3.top')
        self.assertRaises(LammpsError, lambda: load_file(fn))
        self.assertRaises(ValueError, lambda: load_file(fn, defines=dict()))
        f = StringIO('\n[ defaults ]\n1 1 yes\n\n[ system ]\nname\n'
                     '[ molecules ]\nNOMOL  2\n')
        self.assertRaises(LammpsError, lambda: LammpsDataFile(f))
        fn = os.path.join(get_fn('gmxtops'), 'bad_nrexcl.top')
        self.assertRaises(LammpsWarning, lambda:
                LammpsDataFile(fn, defines=dict(SMALL_NREXCL=1))
        )
        self.assertRaises(LammpsWarning, lambda: LammpsDataFile(fn))
        self.assertRaises(LammpsWarning, lambda:
                LammpsDataFile(wfn, defines=dict(NODUP=1))
        )
        self.assertRaises(LammpsError, lambda:
                LammpsDataFile(wfn, defines=dict(NODUP=1, BADNUM=1))
        )
        self.assertRaises(RuntimeError, LammpsDataFile().parametrize)

    def test_data_parsing_missing_types(self):
        """ Test LAMMPS topology files with missing types """
        warnings.filterwarnings('error', category=LammpsWarning)
        self.assertRaises(LammpsWarning, lambda:
                LammpsDataFile(os.path.join(get_fn('gmxtops'),
                                    'missing_atomtype.top'), parametrize=False)
        )
        warnings.filterwarnings('ignore', category=LammpsWarning)
        top = LammpsDataFile(os.path.join(get_fn('gmxtops'),
                                  'missing_atomtype.top'), parametrize=False)
        self.assertIs(top[0].atom_type, UnassignedAtomType)
        self.assertEqual(top[0].mass, -1)
        self.assertEqual(top[0].atomic_number, -1)
        self.assertEqual(top[1].atomic_number, 1)  # taken from atom_type
        self.assertEqual(top[-1].atomic_number, 1) # taken from atom_type
        self.assertEqual(top[-1].charge, 0) # removed
        self.assertEqual(top.bonds[0].funct, 2)
        self.assertTrue(top.unknown_functional)

    def test_lammps_data_write(self):
        """ Tests the LammpsDataFile writer """
        def total_diheds(dlist):
            n = 0
            for d in dlist:
                if isinstance(d.type, DihedralTypeList):
                    n += len(d.type)
                elif not d.improper:
                    n += 1
            return n
        parm = load_file(get_fn('ash.parm7'))
        top = LammpsDataFile.from_structure(parm)
        self.assertRaises(TypeError, lambda: top.write(10))
        f = StringIO()
        self.assertRaises(ValueError, lambda: top.write(f, parameters=10))
        # Write parameters and topology to same filename
        fn = get_fn('test.top', written=True)
        top.write(fn, parameters=fn)
        top2 = load_file(fn)
        self.assertEqual(len(top2.atoms), len(top.atoms))
        self.assertEqual(len(top2.bonds), len(top.bonds))
        self.assertEqual(len(top2.angles), len(top.angles))
        self.assertEqual(total_diheds(top2.dihedrals), total_diheds(top.dihedrals))
        for a1, a2 in zip(top2.atoms, top.atoms):
            self.assertAlmostEqual(a1.atom_type.sigma, a2.atom_type.sigma, places=3)
            self.assertAlmostEqual(a1.atom_type.epsilon, a2.atom_type.epsilon, places=3)
            self.assertEqual(a1.atom_type.name, a2.atom_type.name)
            self.assertEqual(a1.name, a2.name)
            self.assertEqual(a1.type, a2.type)
            self.assertEqual(set(a.name for a in a1.bond_partners),
                             set(a.name for a in a2.bond_partners))
        # Now try passing open files
        with open(fn, 'w') as f:
            top.write(f, parameters=f)
        top2 = load_file(fn)
        self.assertEqual(len(top2.atoms), len(top.atoms))
        self.assertEqual(len(top2.bonds), len(top.bonds))
        self.assertEqual(len(top2.angles), len(top.angles))
        self.assertEqual(total_diheds(top2.dihedrals), total_diheds(top.dihedrals))
        for a1, a2 in zip(top2.atoms, top.atoms):
            self.assertAlmostEqual(a1.atom_type.sigma, a2.atom_type.sigma, places=3)
            self.assertAlmostEqual(a1.atom_type.epsilon, a2.atom_type.epsilon, places=3)
            self.assertEqual(a1.atom_type.name, a2.atom_type.name)
            self.assertEqual(a1.name, a2.name)
            self.assertEqual(a1.type, a2.type)
            self.assertEqual(set(a.name for a in a1.bond_partners),
                             set(a.name for a in a2.bond_partners))
        # Now try separate parameter/topology file
        fn2 = get_fn('test.itp', written=True)
        top.write(fn, parameters=fn2)
        top2 = load_file(fn)
        self.assertEqual(len(top2.atoms), len(top.atoms))
        self.assertEqual(len(top2.bonds), len(top.bonds))
        self.assertEqual(len(top2.angles), len(top.angles))
        self.assertEqual(total_diheds(top2.dihedrals), total_diheds(top.dihedrals))
        for a1, a2 in zip(top2.atoms, top.atoms):
            self.assertAlmostEqual(a1.atom_type.sigma, a2.atom_type.sigma, places=3)
            self.assertAlmostEqual(a1.atom_type.epsilon, a2.atom_type.epsilon, places=3)
            self.assertEqual(a1.atom_type.name, a2.atom_type.name)
            self.assertEqual(a1.name, a2.name)
            self.assertEqual(a1.type, a2.type)
            self.assertEqual(set(a.name for a in a1.bond_partners),
                             set(a.name for a in a2.bond_partners))
        # Now force writing pair types...
        top.defaults.gen_pairs = 'no'
        top.write(fn, parameters=fn)
        top2 = load_file(fn)
        self.assertEqual(len(top2.atoms), len(top.atoms))
        self.assertEqual(len(top2.bonds), len(top.bonds))
        self.assertEqual(len(top2.angles), len(top.angles))
        self.assertEqual(total_diheds(top2.dihedrals), total_diheds(top.dihedrals))
        for a1, a2 in zip(top2.atoms, top.atoms):
            self.assertAlmostEqual(a1.atom_type.sigma, a2.atom_type.sigma, places=3)
            self.assertAlmostEqual(a1.atom_type.epsilon, a2.atom_type.epsilon, places=3)
            self.assertEqual(a1.atom_type.name, a2.atom_type.name)
            self.assertEqual(a1.name, a2.name)
            self.assertEqual(a1.type, a2.type)
            self.assertEqual(set(a.name for a in a1.bond_partners),
                             set(a.name for a in a2.bond_partners))

        # Now force writing pair types to [ pairtypes ] (instead of in-line)
        fn2 = get_fn('testpairtypes.top', written=True)
        top2.write(fn2, parameters=fn2)
        top3 = load_file(fn2)
        self.assertEqual(top3.defaults.gen_pairs, 'no')
        self.assertEqual(len(top2.atoms), len(top3.atoms))
        self.assertEqual(len(top2.bonds), len(top3.bonds))
        self.assertEqual(len(top2.angles), len(top3.angles))
        self.assertEqual(total_diheds(top2.dihedrals), total_diheds(top3.dihedrals))
        for a1, a2 in zip(top2.atoms, top3.atoms):
            self.assertAlmostEqual(a1.atom_type.sigma, a2.atom_type.sigma, places=3)
            self.assertAlmostEqual(a1.atom_type.epsilon, a2.atom_type.epsilon, places=3)
            self.assertEqual(a1.atom_type.name, a2.atom_type.name)
            self.assertEqual(a1.name, a2.name)
            self.assertEqual(a1.type, a2.type)
            self.assertEqual(set(a.name for a in a1.bond_partners),
                             set(a.name for a in a2.bond_partners))
        self.assertEqual(len(top2.adjusts), len(top3.adjusts))
        for adj1, adj2 in zip(top2.adjusts, top3.adjusts):
            self.assertEqual({adj1.atom1.idx, adj1.atom2.idx},
                             {adj2.atom1.idx, adj2.atom2.idx})
            self.assertEqual(adj1.type, adj2.type)

        # We can't combine molecules that don't exist
        self.assertRaises(IndexError, lambda: top.write(fn, combine=[[1, 2]]))
        # Now change all angle types to urey-bradleys and make sure they're
        # written with 0s for those parameters
        psf = load_file(get_fn('ala_ala_ala.psf'))
        psf.load_parameters(
                CharmmParameterSet(get_fn('top_all22_prot.inp'),
                                   get_fn('par_all22_prot.inp'))
        )
        for atom in psf.atoms:
            self.assertIsNot(atom.atom_type, UnassignedAtomType)
        ctop = LammpsDataFile.from_structure(psf)
        for atom in ctop.atoms:
            self.assertIsNot(atom.atom_type, UnassignedAtomType)
            self.assertIsInstance(atom.type, str)
        ctop.write(fn, parameters=fn)
        top2 = load_file(fn)
        self.assertGreater(len(top2.urey_bradleys), 0)
        self.assertEqual(len(top2.urey_bradleys), len(ctop.urey_bradleys))

    _equal_atoms = utils.equal_atoms

@unittest.skipUnless(HAS_LAMMPS, "Cannot run LAMMPS tests without Lammps")
class TestLammpsMissingParameters(FileIOTestCase):
    """ Test handling of missing parameters """

    def setUp(self):
        self.top = load_file(get_fn('ildn.solv.top'), parametrize=False)
        warnings.filterwarnings('error', category=LammpsWarning)
        FileIOTestCase.setUp(self)

    def tearDown(self):
        warnings.filterwarnings('always', category=LammpsWarning)
        FileIOTestCase.tearDown(self)

    def test_missing_pairtypes(self):
        """ Tests handling of missing pairtypes parameters """
        self.top.defaults.gen_pairs = 'no'
        self.assertRaises(ParameterError, self.top.parametrize)

    def test_missing_bondtypes(self):
        """ Tests handling of missing bondtypes parameters """
        b1 = self.top.bonds[0]
        del self.top.parameterset.bond_types[(b1.atom1.type, b1.atom2.type)]
        del self.top.parameterset.bond_types[(b1.atom2.type, b1.atom1.type)]
        self.assertRaises(ParameterError, self.top.parametrize)

    def test_extra_pairs(self):
        """ Tests warning if "extra" exception pair found """
        self.top.adjusts.append(NonbondedException(self.top[0], self.top[-1]))
        self.assertRaises(LammpsWarning, self.top.parametrize)

    def test_missing_angletypes(self):
        """ Tests handling of missing angletypes parameters """
        a1 = self.top.angles[0]
        key = (a1.atom1.type, a1.atom2.type, a1.atom3.type)
        del self.top.parameterset.angle_types[key]
        if key != tuple(reversed(key)):
            del self.top.parameterset.angle_types[tuple(reversed(key))]
        self.assertRaises(ParameterError, self.top.parametrize)

    def test_missing_wildcard_dihedraltypes(self):
        """ Tests handling of wild-card dihedraltypes parameters """
        def get_key(d, wc=None):
            if wc is None:
                return (d.atom1.type, d.atom2.type, d.atom3.type, d.atom4.type)
            if wc == 0:
                return ('X', d.atom2.type, d.atom3.type, d.atom4.type)
            if wc == 3:
                return (d.atom1.type, d.atom2.type, d.atom3.type, 'X')
            else:
                return ('X', d.atom2.type, d.atom3.type, 'X')
        d1 = self.top.dihedrals[0]
        for d in self.top.dihedrals:
            if get_key(d) == get_key(d1): continue
            if get_key(d, 0) == get_key(d1, 0): continue
            if get_key(d, 3) == get_key(d1, 3): continue
            if get_key(d, 0) == get_key(d1, 3): continue
            if get_key(d1, 0) == get_key(d, 3): continue
            if d.improper: continue
            if d.type is not None: continue
            d2 = d
            break
        else:
            assert False, 'Bad test parm'
        # Now make sure the two dihedrals match where only one wild-card is
        # present
        params = self.top.parameterset
        if get_key(d1) in params.dihedral_types:
            del params.dihedral_types[get_key(d1)]
            del params.dihedral_types[tuple(reversed(get_key(d1)))]
        if get_key(d2) in params.dihedral_types:
            del params.dihedral_types[get_key(d2)]
            del params.dihedral_types[tuple(reversed(get_key(d2)))]
        dt1 = DihedralTypeList([DihedralType(10, 180, 1)])
        dt2 = DihedralTypeList([DihedralType(11, 0, 2)])
        params.dihedral_types[get_key(d1, 0)] = dt1
        params.dihedral_types[tuple(reversed(get_key(d1, 0)))] = dt1
        params.dihedral_types[get_key(d2, 3)] = dt2
        params.dihedral_types[tuple(reversed(get_key(d2, 3)))] = dt2

        self.top.parametrize()
        self.assertEqual(d1.type, dt1)
        self.assertEqual(d2.type, dt2)

    def test_missing_dihedraltypes(self):
        """ Tests handling of missing dihedraltypes parameters """
        def get_key(d, wc=None):
            if wc is None:
                return (d.atom1.type, d.atom2.type, d.atom3.type, d.atom4.type)
            if wc == 0:
                return ('X', d.atom2.type, d.atom3.type, d.atom4.type)
            if wc == 3:
                return (d.atom1.type, d.atom2.type, d.atom3.type, 'X')
            else:
                return ('X', d.atom2.type, d.atom3.type, 'X')
        for d in self.top.dihedrals:
            if d.type is not None: continue
            break
        params = self.top.parameterset
        if get_key(d) in params.dihedral_types:
            del params.dihedral_types[get_key(d)]
            del params.dihedral_types[tuple(reversed(get_key(d)))]
        if get_key(d, wc=100) in params.dihedral_types:
            del params.dihedral_types[get_key(d, wc=100)]
            del params.dihedral_types[tuple(reversed(get_key(d, wc=100)))]
        self.assertRaises(ParameterError, self.top.parametrize)

    def test_missing_impropertypes(self):
        """ Tests handling of missing improper type """
        for key in set(self.top.parameterset.improper_periodic_types.keys()):
            del self.top.parameterset.improper_periodic_types[key]
        self.assertRaises(ParameterError, self.top.parametrize)

    def test_wildcard_rbtorsions(self):
        """ Tests handling of missing and wild-cards with R-B torsion types """
        def get_key(d, wc=None):
            if wc is None:
                return (d.atom1.type, d.atom2.type, d.atom3.type, d.atom4.type)
            if wc == 0:
                return ('X', d.atom2.type, d.atom3.type, d.atom4.type)
            if wc == 3:
                return (d.atom1.type, d.atom2.type, d.atom3.type, 'X')
            else:
                return ('X', d.atom2.type, d.atom3.type, 'X')
        for i, d1 in enumerate(self.top.dihedrals):
            if not d1.improper and d1.type is None:
                break
        else:
            assert False, 'Bad topology file for test'
        del self.top.dihedrals[i]
        for i, d2 in enumerate(self.top.dihedrals):
            if get_key(d1) == get_key(d2): continue
            if get_key(d1, 0) == get_key(d2, 0): continue
            if get_key(d1, 3) == get_key(d2, 3): continue
            if get_key(d1, 0) == get_key(d2, 3): continue
            if get_key(d2, 0) == get_key(d1, 3): continue
            if not d2.improper and d2.type is None:
                break
        else:
            assert False, 'Bad topology file for test'
        del self.top.dihedrals[i]
        self.top.rb_torsions.extend([d1, d2])
        self.assertRaises(ParameterError, self.top.parametrize)
        # Now assign wild-cards
        params = self.top.parameterset
        rbt = RBTorsionType(1, 2, 3, 4, 5, 6)
        params.rb_torsion_types[get_key(d1, 0)] = rbt
        params.rb_torsion_types[tuple(reversed(get_key(d1, 0)))] = rbt
        rbt2 = RBTorsionType(2, 3, 4, 5, 6, 7)
        params.rb_torsion_types[get_key(d2, 1)] = rbt2
        params.rb_torsion_types[tuple(reversed(get_key(d2, 1)))] = rbt2

        self.top.parametrize()

        self.assertEqual(d1.type, rbt)
        self.assertEqual(d2.type, rbt2)

    def test_missing_impropers(self):
        """ Test handling of missing impropers """
        self.top.impropers.append(Improper(*tuple(self.top.atoms[:4])))
        self.assertRaises(ParameterError, self.top.parametrize)

    def test_missing_cmaps(self):
        """ Test handling of missing cmaptypes """
        self.top.cmaps.append(Cmap(*tuple(self.top.atoms[:5])))
        self.assertRaises(ParameterError, self.top.parametrize)

    def test_missing_ureybradleys(self):
        """ Test handling of missing Urey-Bradley types """
        self.top.angles[0].funct = 5
        self.top.urey_bradleys.append(
                UreyBradley(self.top.angles[0].atom1, self.top.angles[0].atom3)
        )
        self.assertRaises(ParameterError, self.top.parametrize)

class TestLammpsDataHelperFunctions(FileIOTestCase):
    """ Test LAMMPS helper functions """

    def setUp(self):
        self.top = LammpsDataFile()
        self.top.add_atom(Atom(name='C1'), 'ABC', 1)
        self.top.add_atom(Atom(name='C1'), 'DEF', 2)
        self.top.add_atom(Atom(name='C1'), 'GHI', 3)
        self.top.add_atom(Atom(name='C1'), 'JKL', 4)
        self.top.add_atom(Atom(name='C1'), 'MNO', 5)
        self.top.add_atom(Atom(name='C1'), 'PQR', 5)
        warnings.filterwarnings('error', category=LammpsWarning)
        FileIOTestCase.setUp(self)

    def test_parse_pairs(self):
        """ Test LammpsDataFile._parse_pairs """
        self.assertRaises(LammpsWarning, lambda:
                self.top._parse_pairs('1  2  3\n', dict(), self.top.atoms))
        warnings.filterwarnings('ignore', category=LammpsWarning)
        self.top._parse_pairs('1  2  3\n', dict(), self.top.atoms)
        self.assertTrue(self.top.unknown_functional)

    def test_parse_angles(self):
        """ Test LammpsDataFile._parse_angles """
        self.assertRaises(LammpsWarning, lambda:
                self.top._parse_angles('1  2  3  2\n', dict(), dict(),
                                       self.top.atoms)
        )
        warnings.filterwarnings('ignore', category=LammpsWarning)
        self.top._parse_angles('1  2  3  2\n', dict(), dict(), self.top.atoms)
        self.assertTrue(self.top.unknown_functional)

    def test_parse_dihedrals(self):
        """ Test LammpsDataFile._parse_dihedrals """
        self.assertRaises(LammpsWarning, lambda:
                self.top._parse_dihedrals('1 2 3 4 100\n', dict(), dict(),
                                          self.top)
        )
        warnings.filterwarnings('ignore', category=LammpsWarning)
        self.top._parse_dihedrals('1 2 3 4 100\n', dict(), dict(), self.top)
        self.assertTrue(self.top.unknown_functional)
        self.assertEqual(len(self.top.dihedrals), 1)
        dih = self.top.dihedrals[0]
        self.assertIs(dih.atom1, self.top[0])
        self.assertIs(dih.atom2, self.top[1])
        self.assertIs(dih.atom3, self.top[2])
        self.assertIs(dih.atom4, self.top[3])
        self.assertIs(dih.type, None)
        # Test multi-term dihedrals
        dt = dict()
        PMD = dict()
        self.top._parse_dihedrals('1 2 3 4 9 180 50 1', dt, PMD, self.top)
        self.assertIn(tuple(self.top.atoms[:4]), PMD)
        self.top._parse_dihedrals('1 2 3 4 9 180 40 2', dt, PMD, self.top)
        self.assertEqual(len(PMD[tuple(self.top.atoms[:4])]), 2)

    def test_parse_cmaps(self):
        """ Test LammpsDataFile._parse_cmaps """
        self.assertRaises(LammpsWarning, lambda:
                self.top._parse_cmaps('1 2 3 4 5 2\n', self.top.atoms))
        warnings.filterwarnings('ignore', category=LammpsWarning)
        self.top._parse_cmaps('1 2 3 4 5 2\n', self.top.atoms)
        self.assertTrue(self.top.unknown_functional)

    def test_parse_settles(self):
        """ Test LammpsDataFile._parse_settles """
        self.assertRaises(LammpsError, lambda:
                self.top._parse_settles('whatever', self.top.atoms))
        self.assertRaises(LammpsError, lambda:
                self.top._parse_settles('whatever', self.top.atoms[:3]))
        self.top[0].atomic_number = 8
        self.top[1].atomic_number = self.top[2].atomic_number = 1
        self.assertRaises(LammpsError, lambda:
                self.top._parse_settles('1 2 nofloat nofloat\n',
                                        self.top.atoms[:3])
        )

    def test_parse_vsite3(self):
        """ Test LammpsDataFile._parse_vsites3 """
        self.assertRaises(LammpsError, lambda:
                self.top._parse_vsites3('1 2 3 4 1 1.2 1.3\n', self.top.atoms,
                                        ParameterSet())
        )
        self.assertRaises(LammpsError, lambda:
                self.top._parse_vsites3('1 2 3 4 2 1.2 1.3\n', self.top.atoms,
                                        ParameterSet())
        )
        bond = Bond(self.top[0], self.top[1])
        self.assertRaises(LammpsError, lambda:
                self.top._parse_vsites3('1 2 3 4 1 1.2 1.2', self.top.atoms,
                                        ParameterSet())
        )

    def test_parse_atomtypes(self):
        """ Test LammpsDataFile._parse_atomtypes """
        name, typ = self.top._parse_atomtypes('CX 12.01 0 A 0.1 2.0')
        self.assertEqual(name, 'CX')
        self.assertEqual(typ.atomic_number, 6)
        self.assertEqual(typ.charge, 0)
        self.assertEqual(typ.epsilon, 2.0/4.184)
        self.assertEqual(typ.sigma, 1)

    def test_parse_bondtypes(self):
        """ Test LammpsDataFile._parse_bondtypes """
        self.assertRaises(LammpsWarning, lambda:
                self.top._parse_bondtypes('CA CB 2 0.1 5000'))
        warnings.filterwarnings('ignore', category=LammpsWarning)
        self.top._parse_bondtypes('CA CB 2 0.1 5000')
        self.assertTrue(self.top.unknown_functional)

    def test_parse_angletypes(self):
        """ Test LammpsDataFile._parse_angletypes """
        self.assertRaises(LammpsWarning, lambda:
                self.top._parse_angletypes('CA CB CC 2 120 5000'))
        warnings.filterwarnings('ignore', category=LammpsWarning)
        self.top._parse_angletypes('CA CB CC 2 120 5000')
        self.assertTrue(self.top.unknown_functional)

    def test_parse_dihedraltypes(self):
        """ Test LammpsDataFile._parse_dihedraltypes """
        key, dtype, ptype, replace = self.top._parse_dihedraltypes(
                                        'CA CA 9 180 50.0 2')
        self.assertEqual(key, ('X', 'CA', 'CA', 'X'))
        self.assertEqual(dtype, 'normal')
        self.assertFalse(replace)
        self.assertEqual(ptype.phase, 180)
        self.assertEqual(ptype.phi_k, 50/4.184)
        self.assertEqual(ptype.per, 2)
        self.assertRaises(LammpsWarning, lambda:
                self.top._parse_dihedraltypes('CX CA CA CX 10 180 50.0 2'))
        warnings.filterwarnings('ignore', category=LammpsWarning)
        self.top._parse_dihedraltypes('CX CA CA CX 10 180 50.0 2')
        self.assertTrue(self.top.unknown_functional)

    def test_parse_cmaptypes(self):
        """ Test LammpsDataFile._parse_cmaptypes """
        self.assertRaises(LammpsError, lambda:
                self.top._parse_cmaptypes('C1 C2 C3 C4 C5 1 24 24 1 2 3 4 5'))
        self.assertRaises(LammpsError, lambda:
                self.top._parse_cmaptypes('C1 C2 C3 C4 C5 1 2 3 1 2 3 4 5 6'))

class TestLammpsInput(FileIOTestCase):
    """ Tests the Lammps input file parser """

    def test_input_detection(self):
        """ Tests automatic detection of LAMMPS input files """
        fn = get_fn('candidate.gro', written=True)
        with open(fn, 'w') as f:
            f.write('Some title\n 1000\n    aISNot a valid format\n')
        self.assertFalse(LammpsInputFile.id_format(fn))
        self.assertRaises(LammpsError, lambda: LammpsInputFile.parse(fn))
        f = StringIO('Lammps title line\n notanumber\nsome line\n')
        self.assertRaises(LammpsError, lambda: LammpsInputFile.parse(f))

    def test_read_input_file(self):
        """ Tests reading input file """
        gro = LammpsInputFile.parse(get_fn('1aki.ff99sbildn.gro'))
        self.assertIsInstance(gro, Structure)
        self.assertEqual(len(gro.atoms), 1960)
        self.assertEqual(len(gro.residues), 129)
        self.assertAlmostEqual(gro.atoms[0].xx, 44.6)
        self.assertAlmostEqual(gro.atoms[0].xy, 49.86)
        self.assertAlmostEqual(gro.atoms[0].xz, 18.10)
        self.assertAlmostEqual(gro.atoms[1959].xx, 50.97)
        self.assertAlmostEqual(gro.atoms[1959].xy, 39.80)
        self.assertAlmostEqual(gro.atoms[1959].xz, 38.64)
        self.assertAlmostEqual(gro.box[0], 74.1008)
        self.assertAlmostEqual(gro.box[1], 74.10080712)
        self.assertAlmostEqual(gro.box[2], 74.10074585)
        self.assertAlmostEqual(gro.box[3], 70.52882666)
        self.assertAlmostEqual(gro.box[4], 109.47126278)
        self.assertAlmostEqual(gro.box[5], 70.52875398)
        # Check atomic number and mass assignment
        self.assertEqual(gro.atoms[0].atomic_number, 7)
        self.assertEqual(gro.atoms[0].mass, 14.0067)
        fn = get_fn('test.gro', written=True)
        # Test bad input files
        with open(fn, 'w') as wf, open(get_fn('1aki.charmm27.solv.gro')) as f:
            for i in range(1000):
                wf.write(f.readline())
        self.assertRaises(LammpsError, lambda: LammpsInputFile.parse(fn))
        with open(get_fn('1aki.ff99sbildn.gro')) as f:
            lines = f.readlines()
        lines[-1] = 'not a legal box line\n'
        with open(fn, 'w') as f:
            f.write(''.join(lines))
        self.assertRaises(LammpsError, lambda: LammpsInputFile.parse(fn))

    def test_write_input_file(self):
        """ Tests writing input file """
        gro = LammpsInputFile.parse(get_fn('1aki.ff99sbildn.gro'))
        LammpsInputFile.write(gro, get_fn('1aki.ff99sbildn.gro', written=True))
        gro = load_file(get_fn('1aki.ff99sbildn.gro', written=True))
        self.assertIsInstance(gro, Structure)
        self.assertEqual(len(gro.atoms), 1960)
        self.assertEqual(len(gro.residues), 129)
        self.assertAlmostEqual(gro.atoms[0].xx, 44.6)
        self.assertAlmostEqual(gro.atoms[0].xy, 49.86)
        self.assertAlmostEqual(gro.atoms[0].xz, 18.10)
        self.assertAlmostEqual(gro.atoms[1959].xx, 50.97)
        self.assertAlmostEqual(gro.atoms[1959].xy, 39.80)
        self.assertAlmostEqual(gro.atoms[1959].xz, 38.64)
        self.assertAlmostEqual(gro.box[0], 74.1008)
        self.assertAlmostEqual(gro.box[1], 74.10080712)
        self.assertAlmostEqual(gro.box[2], 74.10074585)
        self.assertAlmostEqual(gro.box[3], 70.52882666)
        self.assertAlmostEqual(gro.box[4], 109.47126278)
        self.assertAlmostEqual(gro.box[5], 70.52875398)
        self.assertRaises(TypeError, lambda: LammpsInputFile.write(gro, 10))


if __name__ == '__main__':
    import parmed as pmd
    struc = pmd.load_file('files/silica_water.data', parametrize=False)
    # assert False
    struc.save('silica_water_conv.data', overwrite=True)
