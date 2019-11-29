import copy
import numpy as np
from itertools import product


def ss( r, sigma ):
    return sigma

def sp( r, sigma, xyz = 'x' ):
    # c = cos\theta = e_xyz \cdot ( R - 0 ) / ( |e_xyz| | R - 0 | )
    epsilon = 1e-7
    if xyz == 'x':
        c = r[0]
    elif xyz == 'y':
        c = r[1]
    else:  # xyz == 'z'
        c = r[2]
    c /= np.linalg.norm( r ) + epsilon
    return c * sigma

def ps( r, sigma, xyz = 'x' ):
    return SlaterKoster_sp( -r, sigma, xyz )

def sd( r, sigma, xyz = 'xy' ):
    epsilon = 1e-7
    norm = np.linalg.norm( r ) + epsilon
    l = r[0] / norm
    m = r[1] / norm
    n = r[0] / norm

    if xyz in ( 'xy', 'yz', 'zx', 'yx', 'zy', 'xz' ):

        if   xyz in ( 'xy', 'yx' ):
            c = l * m
        elif xyz in ( 'yz', 'zy' ):
            c = m * n 
        else:  # xyz in ( 'zx', 'xz' )
            c = n * l
        return np.sqrt(3) * c * sigma

    elif xyz == 'x2':
        return np.sqrt(3)/2 * ( l**2 - m**2 ) * sigma

    elif xyz in ( '3z', 'z2' ):
        return ( n**2 - ( l**2 + m**2 )/2 ) * sigma

    else:
        print( 'check xyz for SlaterKoster.sd' )
        return False

def ds( r, sigma, xyz = 'xy' ):
    return sd( -r, sigma, xyz )

def pp( r, sigma, pi, xyz1 = 'x', xyz2 = 'x' ):
    epsilon = 1e-7
    if not xyz1 in ( 'x', 'y', 'z' ):
        return False
    if not xyz2 in ( 'x', 'y', 'z' ):
        return False
    if xyz1 == xyz2:
        if xyz1 == 'x':
            c = r[0]
        elif xyz1 == 'y':
            c = r[1]
        else:  # xyz1 == xyz2 == 'z'
            c = r[2]
        c /= np.linalg.norm( r ) + epsilon
        return c**2 * sigma + ( 1 - c**2 ) * pi
    else:
        if xyz1 == 'x':
            c1 = r[0]
            if xyz2 == 'y':
                c2 = r[1]
            else:  # xyz2 == 'z'
                c2 = r[2]
        elif xyz2 == 'y':
            c1 = r[1]
            if xyz2 == 'z':
                c2 = r[2]
            else:  # xyz2 == 'x'
                c2 = r[0]
        else:  # xyz1 == 'z'
            c1 = r[2]
            if xyz2 == 'x':
                c2 = r[0]
            else:  # xyz2 == 'y'
                c2 = r[1]
        c = c1 * c2 / ( np.linalg.norm( r ) + epsilon )**2
        return c * ( sigma - pi )

def dd( r, sigma, pi, delta, xyz1 = 'xy', xyz2 = 'xy' ):
    epsilon = 1e-7
    norm = np.linalg.norm( r ) + epsilon
    lmn  = np.array([ r_ / norm for r_ in r ])
    lmn2 = lmn**2
    tuple_xy = ( 'xy', 'yx' )
    tuple_yz = ( 'yz', 'zy' )
    tuple_zx = ( 'zx', 'xz' )

    group1 = ( 'xy', 'yz', 'zx', 'yx', 'zy', 'xz' )

    if   xyz1 in tuple_xy:
        ( i1, j1, k1 ) = ( 0, 1, 2 )
    elif xyz1 in tuple_yz:
        ( i1, j1, k1 ) = ( 2, 0, 1 )
    else:  # xyz1 in tuple_zx
        ( i1, j1, k1 ) = ( 1, 2, 0 )
    
    if   xyz2 in tuple_xy:
        ( i2, j2, k2 ) = ( 0, 1, 2 )
    elif xyz2 in tuple_yz:
        ( i2, j2, k2 ) = ( 2, 0, 1 )
    else:  # xyz2 in tuple_zx
        ( i2, j2, k2 ) = ( 1, 2, 0 )

    if ( xyz1, xyz2 ) in product( group1, repeat = 2 ):
        
        if ( xyz1, xyz2 ) in ( ( xyz_, xyz_ ) for xyz_ in group1 ):
        
            result  = 3 * lmn2[ i1 ] * lmn2[ j1 ] * sigma
            result += ( lmn2[ i1 ] + lmn2[ j1 ] + 4 * lmn2[ i1 ] * lmn2[ j1 ] ) * pi
            result += ( lmn2[ k1 ] + lmn2[ i1 ] * lmn2[ j1 ] ) * delta
            return result
    
        else:  # for example: ( 'xy', 'yz' )

            if     ( xyz1, xyz2 ) in ( ( xy_, zx_ ) for xy_ in tuple_xy for zx_ in tuple_zx ) \
               or  ( xyz1, xyz2 ) in ( ( yz_, xy_ ) for yz_ in tuple_yz for xy_ in tuple_xy ) \
               or  ( xyz1, xyz2 ) in ( ( zx_, yz_ ) for zx_ in tuple_zx for yz_ in tuple_yz ):

                ( i_, j_, k_ ) = copy.deepcopy( ( i1, j1, k1 ) )
                ( i1, j1, k1 ) = copy.deepcopy( ( i2, j2, k2 ) )
                ( i2, j2, k2 ) = copy.deepcopy( ( i_, j_, k_ ) )
            
            result  = 3 * lmn[ i1 ] * lmn[ j1 ] * lmn[ i2 ] * lmn[ j2 ] * sigma
            result += lmn[ i1 ] * lmn[ j2 ] * ( 1 - 4 * lmn[ j1 ] * lmn[ i2 ] ) * pi
            result += lmn[ i1 ] * lmn[ j2 ] * ( lmn[ j1 ] * lmn[ i2 ] - 1 ) * delta
            return result


def sk( r, V, l1, l2 ):
    # V = ( sigma, pi, delta )

    tuple_p  = ( 'px', 'py', 'pz' )
    tuple_d  = ( 'dxy', 'dyz', 'dzx', 'dyx', 'dzy', 'dxz', 'd3z', 'dz2', 'dx2' )

    L = ( l1, l2 )

    if   L == ( 's', 's' ):
        return ss( r, V[0] )
    
    elif L in [ ( 's', p_ ) for p_ in tuple_p ] + [ ( p_, 's' ) for p_ in tuple_p ]:
        r_ = r
        if l1 in tuple_p:
            r_ = -r
            return sp( r_, V[0], l1[1] )
        else:
            return sp( r_, V[0], l2[1] )
    
    elif L in product( tuple_p, repeat = 2 ):
        return pp( r, V[0], V[1], l1[1], l2[1] )
    
    elif L in [ ( 's', d_ ) for d_ in tuple_d ] + [ ( d_, 's ') for d_ in tuple_d ]:
        r_ = r
        if l1 in tuple_d:
            r_ = -r
            return sd( r_, V[0], l1[1:3] )
        else:
            return sd( r_, V[0], l2[1:3] )

    elif L in product( tuple_d, repeat = 2 ):
        return dd( r, V[0], V[1], V[2], l1[1:3], l2[1:3] )
