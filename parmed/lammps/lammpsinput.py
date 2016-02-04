"""
This module contains functionality relevant to loading and parsing LAMMPS input
(configuration) files and building an unparametrized Structure containing only
forcefield information.
"""

from __future__ import print_function, division, absolute_import

from contextlib import closing
import warnings

from parmed.exceptions import LammpsError, LammpsWarning
from parmed.formats.registry import FileFormatType
from parmed.periodic_table import AtomicNum, element_by_name, Mass
from parmed.structure import Structure
from parmed import unit as u
from parmed.utils.io import genopen
from parmed.utils.six import add_metaclass, string_types


@add_metaclass(FileFormatType)
class LammpsInputFile(object):
    """ Parses and writes LAMMPS input files """
    #===================================================

    @staticmethod
    def id_format(filename):
        """ Identifies the file as a LAMMPS input file

        Parameters
        ----------
        filename : str
            Name of the file to check if it is a LAMMPS input file

        Returns
        -------
        is_fmt : bool
            If it is identified as a LAMMPS input file, return True. False
            otherwise
        """
        with closing(genopen(filename)) as f:
            return False

    #===================================================

    def read(self, filename):
        """ Read a LAMMPS input file into a Structure

        Parameters
        ----------
        filename : str or file-like
            Name of the file or the input file object

        Returns
        -------
        struct : :class:`Structure`
            The Structure instance instantiated with *just* residues and atoms
            populated (with coordinates)
        """
        struct = Structure()
        parsable_keywords = {
            'units': self.parse_units,
            'atom_style': self.parse_atom_style,
            'dimension': self.parse_dimension,
            'boundary': self.parse_boundary,
            'pair_style': self.parse_pair_style,
            'kspace_style': self.parse_kspace_style,
            'pair_modify': self.parse_pair_modify,
            'bond_style': self.parse_bond_style,
            'angle_style': self.parse_angle_style,
            'dihedral_style': self.parse_dihedral_style,
            'improper_style': self.parse_improper_style,
            'special_bonds': self.parse_special_bonds,
            'read_data': self.parse_read_data}

        defaults = [
            'units lj',
            'atom_style atomic',
            'dimension 3',
            'boundary p p p',
            'pair_style none',
            'kspace_style none',
            'pair_modify mix geometric shift no table 12 tabinner sqrt(2.0) tail no compute yes',
            'bond_style none',
            'angle_style none',
            'dihedral_style none',
            'improper_style none',
            'special_bonds lj 0.0 0.0 0.0 coul 0.0 0.0 0.0 angle no dihedral no extra 0']

        keyword_defaults = {x.split()[0]: x for x in defaults}
        keyword_check = {x: False for x in keyword_defaults.keys()}

        if isinstance(filename, string_types):
            fileobj = genopen(filename, 'r')
            own_handle = True
        else:
            fileobj = filename
            own_handle = False
        try:
            for line in fileobj:
                if line.strip():
                    keyword = line.split()[0]
                    if keyword in parsable_keywords:
                        parsable_keywords[keyword](line.split())
                        keyword_check[keyword] = True
        finally:
            for key in keyword_check.keys():
                if not keyword_check[key]:
                    warnings.warn('Keyword {0} not set, using LAMMPS default value {1}'.format(key, " ".join(keyword_defaults[key].split()[1:])))
                    parsable_keywords[key](keyword_defaults[key].split())

            self.set_units(self.unit_set)
            if own_handle:
                fileobj.close()

        return struct

    #===================================================

    @staticmethod
    def write(struct, dest):
        """ Write a LAMMPS input file from a Structure

        Parameters
        ----------
        struct : :class:`Structure`
            The structure to write to a LAMMPS input file
        dest : str or file-like
            The name of a file or a file object to write the LAMMPS input file to

        """
        warnings.warn('Writing of LAMMPS input files is not yet supported',
                      LammpsWarning)
        if isinstance(dest, string_types):
            dest = genopen(dest, 'w')
            own_handle = True
        elif not hasattr(dest, 'write'):
            raise TypeError('dest must be a file name or file-like object')

        dest.write('NOT IMPLEMENTED')

        if own_handle:
            dest.close()
