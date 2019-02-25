import math
import numpy as np
from ase.spacegroup import crystal
from scipy import interpolate

def ChemicalShift( x, y, shift ):

    x_min = x[  0 ]
    x_max = x[ -1 ]
    f     = interpolate.interp1d( x, y, kind = 'cubic' )

    x_shifted = [ x_ - shift for x_ in x ]
    y_cs      = []

    for xs in x_shifted:
        if x_min <= xs <= x_max:
            y_cs.append( f( xs ) )
        else:
            y_cs.append( 0. )

    return y_cs

def StructuralCluster( system, coord, sg, lc, radius ): 

    n_unit_cell_a    = math.ceil( 2. * radius / lc[ 0 ] )
    n_unit_cell_b    = math.ceil( 2. * radius / lc[ 1 ] )
    n_unit_cell_c    = math.ceil( 2. * radius / lc[ 2 ] )

    # make bulk structure
    unit_cell  = crystal( system,
                         coord,
                         spacegroup = sg,
                         cellpar = lc     )
    super_cell = unit_cell.repeat( ( n_unit_cell_a,
                                     n_unit_cell_b,
                                     n_unit_cell_c  ) )

    # cut it to make cluster sphere
    vec_center_of_mass = super_cell.get_center_of_mass()
    super_cell.translate( vec_center_of_mass )
    super_cell.wrap()
    distances = super_cell.get_all_distances()[ 0 ]
    del super_cell[ [ i for i, dis in enumerate( distances )
                      if dis > radius                        ] ]
    super_cell.translate( - vec_center_of_mass )

    return super_cell
