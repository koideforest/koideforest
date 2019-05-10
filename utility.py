import math
import numpy as np
from ase.spacegroup import crystal
from scipy import interpolate
from scipy.fftpack import fft, ifft, fftfreq, fftshift

def FourierTransform( t, s ):
    # tilde{f}( k      ) = \int dx e^{ - i k      x } f( x )
    # or
    # tilde{s}( \omega ) = \int dt e^{ - i \omega t } s( t )
    #                    ~ \Delta_t FFT( s )

    # b[n] = \sum^{N/2}_{m=-N/2} e^{ - 2\pi i n m / N } a[m]

    # 2\pi n m / N = ( 2\pi n / ( N \Delta ) ) ( \Delta m )
    #              = k_n x_m = \omega t

    N = len( t )
    delta_t = t[1] - t[0]
    fft_s   = fft( s )
    fft_s   = fftshift( fft_s )
    tilde_s = delta_t * fft_s

    # fftfreq: [ 0, 1, ..., N/2-1, -N/2, ..., -1 ] / (N \Delta)
    omega   = fftfreq( N, delta_t ) * 2. * np.pi
    omega   = fftshift( omega )
    return omega, tilde_s, fft_s

def IFourierTransform( omega, tilde_s ):
    # f( x ) = (1/(2\pi)) \int dk      e^{ i k      x } \tilde f( k      )
    # or
    # s( t ) = (1/(2\pi)) \int d\omega e^{ i \omega t } \tilde s( \omega )
    #        ~ (N/(2\pi)) \Delta_\omega IFFT( \tilde s )
    #      ( ~ (N/(2\pi)) \Delta_\omega \Delta_t IFFT( fft_s ) = IFFT( fft_s ) )

    # Delta_\omega = 2\pi / ( N \Delta_t ) = 2\pi / T
    # Delta_t      = T / N 

    # a[m] = (1/N) \sum^{N/2}_{n=-N/2} e^{ - 2\pi i n m / N } b[n]

    N       = len( omega )
    delta_omega = omega[1] - omega[0]

    # Caution: omega and tilde_s were already operated by fftshift!
    # fftshift( fftshift( a ) ) = a if N is even.
    s  = ifft( fftshift( tilde_s ) )
    s *= N / (2.*np.pi) * delta_omega  # "s" does not need fftshift

    # fftfreq: [ 0, 1, ..., N/2-1, -N/2, ..., -1 ] / (N \Delta)
    # 1 / (N \Delta_omega) = \Delta_t / 2\pi
    t = fftshift( fftfreq( N, delta_omega ) * 2.*np.pi ) 
    return t, s

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
