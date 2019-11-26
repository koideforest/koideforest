import itertools
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter

class ReciprocalVector_band():

    def __init__( self, start = [], label = '' ):
        if start == []:
            self.k = []
        else:
            self.k = [ np.array( start ) ]
        if label == '':
            self.ticklabel = []
            self.tick      = []
            self.axis      = []
        else:
            self.ticklabel = [ label ]
            self.tick      = [ 0 ]
            self.axis      = [ 0 ]
        self.mesh = 100

    def initial_k_point( self, start, label = '' ):
        self.k.append( start )
        self.ticklabel.append( label )
        try:
            self.tick.append( self.tick[-1] + 1 )
        except IndexError:
            self.tick.append( 0 )
        try:
            self.axis.append( self.axis[-1] + 1 )
        except IndexError:
            self.axis.append( 0 )

    def interpolate_k( self, start, end, n ):
        interval = ( np.array( end ) - np.array( start ) ) / n
        k = []
        for i in range( n ):
            k_ = start + interval * ( i + 1 )
            k.append( k_ )
        return k

    def add_k_point( self, end, label = '', n = -1 ):
        if n == -1:
            n = self.mesh
        self.k += self.interpolate_k( self.k[ -1 ], end, n )
        self.ticklabel.append( label )
        try:
            self.tick.append( self.tick[-1] + n )
        except IndexError:
            self.tick.append( n )
        try:
            self.axis += [ self.axis[-1] + i + 1 for i in range( n ) ]
        except IndexError:
            self.axis += [ i + 1 for i in range( n ) ]

    def reset( self ):
        self.k         = [] 
        self.ticklabel = []
        self.tick      = []
        self.axis      = []


def Neighbor1_SC( a ):
    R = []
    pm = ( -1, 1 )
    for i in pm:
        R.append( a * np.array( [ i, 0, 0 ] ) )
        R.append( a * np.array( [ 0, i, 0 ] ) )
        R.append( a * np.array( [ 0, 0, i ] ) )
    return R

def Neighbor2_SC( a ):
    R = []
    pm = ( -1, 1 )
    for i1 in pm:
        for i2 in pm:
            R.append( a * np.array( [ i1, i2, 0 ] ) )
            R.append( a * np.array( [ 0, i1, i2 ] ) )
            R.append( a * np.array( [ i2, 0, i1 ] ) )
    return R

def Neighbor3_SC( a ):
    R = []
    pm = ( -1, 1 )
    for i1 in pm:
        for i2 in pm:
           for i3 in pm:
            R.append( a * np.array( i1, i2, i3 ) )
    return R

def Neighbor1_BCC( a ):
    return Neighbor3_SC( a/2 )

def Neighbor2_BCC( a ):
    return Neighbor1_SC( a )

def Neighbor1_FCC( a ):
    return Neighbor2_SC( a/2 )

def Neighbor2_FCC( a ):
    return Neighbor1_SC( a )

def Neighbor1_Graphene( LC ):
    # Translational vector
    t1 = LC * np.array( [  1/2, np.sqrt(3)/2, 0 ] )
    t2 = LC * np.array( [ -1/2, np.sqrt(3)/2, 0 ] )
    T  = ( t1, t2 )

    # atomic site in a unit cell 
    zero = np.zeros(3)
    dn   = LC/np.sqrt(3)
    a1   = zero
    a2   = dn * np.array( [ 0, 1, 0 ] )
    A    = ( a1, a2 )

    #pmz = ( -1, 0, 1 )
    #R_ = []
    #for i1 in pmz:
    #    for i2 in pmz:
    #        R_.append( a1 + i1 * t1 + i2 * t2 )
    #        R_.append( a2 + i1 * t1 + i2 * t2 )
    #R = []
    #for a_ in A:
    #    R.append( [ r__ for r__ in R_ if ( np.linalg.norm( r__ - a_ ) <= dn ) and any( r__ - a_ != zero ) ] )
    #return R

    return Neighbor( T, 1, A, dn )

def Neighbor1_Graphene_0( LC ):
    # Translational vector
    t1 = LC * np.array( [  1/2, np.sqrt(3)/2, 0 ] )
    t2 = LC * np.array( [ -1/2, np.sqrt(3)/2, 0 ] )
    
    # atomic site in a unit cell 
    zero = np.zeros(3)
    dn   = LC/np.sqrt(3)
    a1   = zero
    a2   = dn * np.array( [ 0, 1, 0 ] )
    A    = ( a1, a2 )

    pmz = ( -1, 0, 1 )
    R_ = []
    for i1 in pmz:
        for i2 in pmz:
            R_.append( a1 + i1 * t1 + i2 * t2 )
            R_.append( a2 + i1 * t1 + i2 * t2 )

    R = []
    for a_ in A:
        R.append( [ r__ for r__ in R_ if ( np.linalg.norm( r__ - a_ ) <= dn ) and any( r__ - a_ != zero ) ] )
    return R


def Neighbor( T, nt, A, dn ):
    R_  = []
    # for example ( -1, 0, 1 ) -> ( -1, -1 ), ( -1, 0 ), ( -1, 1 )...
    #ijk = itertools.product( np.arange( -nt, nt + 1 ), repeat = len( T ) )

    for a_ in A:
        ijk = itertools.product( np.arange( -nt, nt + 1 ), repeat = len( T ) )
        for i_ in ijk:
            R_.append( np.array( a_ ) + sum( [ i_[it] * np.array( t_ ) for it, t_ in enumerate( T ) ] ) )

    zero = np.zeros( len( T[0] ) ) 
    R    = []
    for a_ in A:
        R.append( [ r__ for r__ in R_ if ( np.linalg.norm( r__ - a_ ) <= dn ) and any( r__ - a_ != zero ) ] )
    return R

def TranslationVector_Neighbor( r0, r1, T, nt, dn ):
    ijk  = itertools.product( np.arange( -nt, nt + 1 ), repeat = len( T ) )
    zero = np.zeros( len( T[0] ) )

    TVN = []
    for i_ in ijk:
        t__ = sum( [ i_[ it ] * np.array( t_ ) for it, t_ in enumerate( T ) ] )
        r_  = np.array( r1 ) + t__
        if ( np.linalg.norm( r_ - np.array( r0 ) ) <= dn ) and any( r_ - np.array( r0 ) != zero ):
            TVN.append( t__ )
    return TVN


def SlaterKoster_ss( r, sigma ):
    return sigma

def SlaterKoster_sp( r, sigma, xyz = 'x' ):
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

def SlaterKoster_ps( r, sigma, xyz = 'x' ):
    return SlaterKoster_sp( -r, sigma, xyz )

def SlaterKoster_pp( r, sigma, pi, xyz1 = 'x', xyz2 = 'x' ):
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

def SlaterKoster( r, V, l1, l2 ):
    # V = ( sigma, pi, delta )
    L = { l1, l2 }
    if   L == { 's', 's' }:
        return SlaterKoster_ss( r, V[0] )
    elif L == { 's', 'px' }:
        r_ = r
        if l1 == 'px':
            r_ = -r
        return SlaterKoster_sp( r_, V[0], 'x' )
    elif L == { 's', 'py' }:
        r_ = r
        if l1 == 'py':
            r_ = -r
        return SlaterKoster_sp( r_, V[0], 'y' )
    elif L == { 's', 'px' }:
        r_ = r
        if l1 == 'pz':
            r_ = -r
        return SlaterKoster_sp( r_, V[0], 'z' )
    elif L == { 'px', 'px' }:
        return SlaterKoster_pp( r, V[0], V[1], 'x', 'x' )
    elif L == { 'py', 'py' }:
        return SlaterKoster_pp( r, V[0], V[1], 'y', 'y' )
    elif L == { 'pz', 'pz' }:
        return SlaterKoster_pp( r, V[0], V[1], 'z', 'z' )
    elif L == { 'px', 'py' }:
        return SlaterKoster_pp( r, V[0], V[1], 'x', 'y' )
    elif L == { 'py', 'pz' }:
        return SlaterKoster_pp( r, V[0], V[1], 'y', 'z' )
    elif L == { 'pz', 'px' }:
        return SlaterKoster_pp( r, V[0], V[1], 'z', 'x' )


if __name__ == '__main__':

    a = 2
    R = Neighbor1_FCC( a )

    #vec_k = ReciprocalVector( list( np.pi/a * np.array( [ 1, 1, 1 ] ) ), 'L' )
    vec_k = ReciprocalVector_band( )
    vec_k.initial_k_point( list( np.pi/a * np.array( [ 1, 1, 1 ] ) ), 'L' )
    vec_k.add_k_point( [ 0 ] * 3, '$\Gamma$' ) 
    vec_k.add_k_point( [ 2 * np.pi / a, 0, 0  ], 'X' )
 
    # s band
    h0    = -1
    sigma = -0.2
    T     = SlaterKoster_ss( R, sigma )
    E     = []
    for k_ in vec_k.k:
        E_ = [ np. exp( 1j * np.dot( k_, r_ ) ) * SlaterKoster_ss( r_, sigma ) for r_ in R ]
        E.append( h0 + sum( E_ ).real )
    
    fig = plt.figure()
    ax  = fig.add_subplot( 1, 1, 1 )
    ax.plot( vec_k.axis, E )
    ax.set_xticks( vec_k.tick )
    ax.set_xticklabels( vec_k.ticklabel )
    plt.show()

    # p band
    h0    = -1
    sigma = 0.2
    pi    = -0.02
    E, U  = [], []
    xyz   = ( 'x', 'y', 'z' )

    for k_ in vec_k.k:
        H = np.zeros( ( 3, 3 ), dtype = np.complex )
        for i1 in range( 3 ):
            for i2 in range( 3 ):
                if i1 == i2:
                    H[ i1 ][ i2 ] += h0
                H[ i1 ][ i2 ] += sum( [   np.exp( 1j * np.dot( k_, r_ ) ) 
                                        * SlaterKoster_pp( r_, sigma, pi, xyz[i1], xyz[i2] ) for r_ in R ] )
        value, vector = np.linalg.eigh( np.real( H ) )
        eigen = sorted( [ [ val, vec ] for val, vec in zip( value, vector ) ], key = itemgetter( 0 ) )
        E.append( [ eig[0] for eig in eigen ] )
        U.append( [ eig[1] for eig in eigen ] )

    E = np.array( E )
    U = np.array( U )

    fig    = plt.figure()
    ax     = fig.add_subplot( 1, 1, 1 )
    colors = ( 'r', 'g', 'b' )
    for e_, u_ in zip( E.T, U.transpose( 1, 2, 0 ) ):  # E[k][n], U[k][n][i]
        for iu, u__ in enumerate( u_ ):
            ax.scatter( vec_k.axis, e_, s = abs( u__ )**2 * 50, c = colors[ iu ], lw = 0 )
    ax.set_xticks( vec_k.tick )
    ax.set_xticklabels( vec_k.ticklabel )
    plt.show()

