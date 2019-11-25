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

def NeighborSC1( a ):
    R = []
    pm = ( -1, 1 )
    for i in pm:
        R.append( a * np.array( [ i, 0, 0 ] ) )
        R.append( a * np.array( [ 0, i, 0 ] ) )
        R.append( a * np.array( [ 0, 0, i ] ) )
    return R

def NeighborSC2( a ):
    R = []
    pm = ( -1, 1 )
    for i1 in pm:
        for i2 in pm:
            R.append( a * np.array( [ i1, i2, 0 ] ) )
            R.append( a * np.array( [ 0, i1, i2 ] ) )
            R.append( a * np.array( [ i2, 0, i1 ] ) )
    return R

def NeighborSC3( a ):
    R = []
    pm = ( -1, 1 )
    for i1 in pm:
        for i2 in pm:
           for i3 in pm:
            R.append( a * np.array( i1, i2, i3 ) )
    return R

def NeighborBCC1( a ):
    return NeighborSC3( a/2 )

def NeighborBCC2( a ):
    return NeighborSC1( a )

def NeighborFCC1( a ):
    return NeighborSC2( a/2 )

def NeighborFCC2( a ):
    return NeighborSC1( a )

def SlaterKoster_ss( R, sigma ):
    T = []
    for r_ in R:
        T.append( sigma )
    return T

def SlaterKoster_sp( R, sigma, xyz = 'x' ):
    # c = cos\theta = e_xyz \cdot ( R - 0 ) / ( |e_xyz| | R - 0 | )
    epsilon = 1e-7
    T = []
    for r_ in R:
        if xyz == 'x':
            c = r_[0]
        elif xyz == 'y':
            c = r_[1]
        else:  # xyz == 'z'
            c = r_[2]
        c /= np.linalg.norm( r_ ) + epsilon
        T.append( c * sigma )
    return T

def SlaterKoster_ps( R, sigma, xyz = 'x' ):
    return SlaterKoster_sp( [ -r_ for r_ in R ], sigma, xyz )

def SlaterKoster_pp( R, sigma, pi, xyz1 = 'x', xyz2 = 'x' ):
    epsilon = 1e-7
    T = []
    if not xyz1 in ( 'x', 'y', 'z' ):
        return T
    if not xyz2 in ( 'x', 'y', 'z' ):
        return T
    for r_ in R:
        if xyz1 == xyz2:
            if xyz1 == 'x':
                c = r_[0]
            elif xyz1 == 'y':
                c = r_[1]
            else:  # xyz1 == xyz2 == 'z'
                c = r_[2]
            c /= np.linalg.norm( r_ ) + epsilon
            T.append( c**2 * sigma +( 1 - c**2 ) * pi )
        else:
            if xyz1 == 'x':
                c1 = r_[0]
                if xyz2 == 'y':
                    c2 = r_[1]
                else:  # xyz2 == 'z'
                    c2 = r_[2]
            elif xyz2 == 'y':
                c1 = r_[1]
                if xyz2 == 'z':
                    c2 = r_[2]
                else:  # xyz2 == 'x'
                    c2 = r_[0]
            else:  # xyz1 == 'z'
                c1 = r_[2]
                if xyz2 == 'x':
                    c2 = r_[0]
                else:  # xyz2 == 'y'
                    c2 = r_[1]
            c = c1 * c2 / ( np.linalg.norm( r_ ) + epsilon )**2
            T.append( c * ( sigma - pi )  )
    return T


if __name__ == '__main__':

    a = 2
    R = NeighborFCC1( a )

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
        E_ = [ np. exp( 1j * np.dot( k_, r_ ) ) * t_ for r_, t_ in zip( R, T ) ]
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
                T = SlaterKoster_pp( R, sigma, pi, xyz[i1], xyz[i2] )
                H[ i1 ][ i2 ] += sum( [ np.exp( 1j * np.dot( k_, r_ ) ) * t_ for r_, t_ in zip( R, T ) ] )
        value, vector = np.linalg.eigh( np.real( H ) )
        eigen = sorted( [ [ val, vec ] for val, vec in zip( value, vector ) ], key = itemgetter( 0 ) )
        E.append( [ eig[0] for eig in eigen ] )
        U.append( [ eig[1] for eig in eigen ] )

    E = np.array( E )
    U = np.array( U )

    fig    = plt.figure()
    ax     = fig.add_subplot( 1, 1, 1 )
    colors = ( 'r', 'g', 'b' )
    for e_, u_ in zip( E.T, U.T ):
        for iu, u__ in enumerate( u_ ):
            ax.scatter( vec_k.axis, e_, s = abs( u__ )**2 * 50, c = colors[ iu ], lw = 0 )
    ax.set_xticks( vec_k.tick )
    ax.set_xticklabels( vec_k.ticklabel )
    plt.show()

