from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as SA

from os.path import basename
from os.path import splitext
from sys import argv
from subprocess import run

if len(argv) < 2:
    raise ValueError

fn_inp = argv[1] 
atoms = ase_read(fn_inp)
structure = AAA.get_structure(atoms)

fn_out = splitext(basename(argv[1]))[0]+'_pc.cif' 
finder = SA(structure)
pc = finder.find_primitive()
pc.to(fn_out) 

run(f"cif2cell {fn_out} -p pwscf -o pw.scf.in", shell=True)
