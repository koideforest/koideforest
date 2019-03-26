import math
import numpy as np
from ase.spacegroup import crystal
from scipy import interpolate
from scipy.fftpack import fft, fftfreq, fftshift

def FourierTransform( t, s ):
    N = len( t )
    delta_t = t[1] - t[0]
    tilde_s = fft( s ) * delta_t
    tilde_s = fftshift( tilde_s )
    omega   = fftfreq( N, delta_t ) * 2. * np.pi
    omega   = fftshift( omega )
    return omega, tilde_s

def ChangeX( x, y, new_x ):

    x_min = x[  0 ]
    x_max = x[ -1 ]
    f     = interpolate.interp1d( x, y, kind = 'cubic' )

    y_cx  = []

    for x_ in new_x:
        if x_min <= x_ <= x_max:
            y_cx.append( f( x_ ) )
        else:
            y_cx.append( 0. )

    return y_cx


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

def ChangeX_ChemicalShift( x, y, new_x, shift ):
    y_cx    = ChangeX( x, y, new_x )
    y_cx_cs = ChemicalShift( new_x, y_cx, shift )
    
def ChemicalShift_ChangeX( x, y, shift, new_x ):
    y_cs    = ChemicalShift( x, y, shift )
    y_cs_cx = ChangeX( x, y_cs, new_x )

def Difference( x, y, x0, y0 ):
    y_cx = ChangeX( x, y, x0 )
    return [ y_cx_ - y0_ for y_cx_, y0_ in zip( y_cx, y0 ) ]

def StructuralCluster( system, coord, sg, lc, radius ): 

    n_unit_cell_a    = math.ceil( 2. * radius / lc[ 0 ] )
    n_unit_cell_b    = math.ceil( 2. * radius / lc[ 1 ] )
    n_unit_cell_c    = math.ceil( 2. * radius / lc[ 2 ] )

    # make bulk structure
    unit_cell  = crystal( system,
                          coord,
                          spacegroup = sg,
                          cellpar    = lc     )
    super_cell = unit_cell.repeat( ( n_unit_cell_a,
                                     n_unit_cell_b,
                                     n_unit_cell_c  ) )

    # cut it to make cluster sphere
    cluster            = super_cell
    vec_center_of_mass = cluster.get_center_of_mass()
    cluster.translate( vec_center_of_mass )
    cluster.wrap()
    distances = cluster.get_all_distances()[ 0 ]
    del cluster[ [ i for i, dis in enumerate( distances )
                      if dis > radius                        ] ]
    cluster.translate( - vec_center_of_mass )

    return cluster
