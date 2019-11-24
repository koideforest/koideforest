import numpy as np
from matplotlib import pyplot as plt

class ReciprocalVector():

    def __init__( self, start = [], label = '' ):
        if start == []:
            self.k = []
        else:
            self.k = [ np.array( start ) ]
        if label == '':
            self.label     = []
            self.labeltick = []
            self.tick      = []
        else:
            self.label     = [ label ]
            self.labeltick = [ 0 ]
            self.tick      = [ 0 ]
        self.mesh = 100

    def initial_k_point( self, start, label = '' ):
        self.k.append( start )
        self.label.append( label )
        try:
            self.labeltick.append( self.labeltick[-1] + 1 )
        except IndexError:
            self.labeltick.append( 0 )
        try:
            self.tick.append( self.tick[-1] + 1 )
        except IndexError:
            self.tick.append( 0 )

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
        self.label.append( label )
        try:
            self.labeltick.append( self.labeltick[-1] + n )
        except IndexError:
            self.labeltick.append( n )
        try:
            self.tick += [ self.tick[-1] + i + 1 for i in range( n ) ]
        except IndexError:
            self.tick += [ i + 1 for i in range( n ) ]

    def reset( self ):
        self.k         = [] 
        self.label     = []
        self.labeltick = []
        self.tick      = []

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

if __name__ == '__main__':

    a = 2
    R = NeighborFCC1( a )
    T = SlaterKoster_ss( sigma )
    
    #vec_k = ReciprocalVector( list( np.pi/a * np.array( [ 1, 1, 1 ] ) ), 'L' )
    vec_k = ReciprocalVector( )
    vec_k.initial_k_point( list( np.pi/a * np.array( [ 1, 1, 1 ] ) ), 'L' )
    vec_k.add_k_point( [ 0 ] * 3, '$\Gamma$' ) 
    vec_k.add_k_point( [ 2 * np.pi / a, 0, 0  ], 'X' )
    
    h0 = -1
    E = []
    for k_ in vec_k.k:
        E_ = [ np. exp( 1j * np.dot( k_, r_ ) ) * t_ for r_, t_ in zip( R, T ) ]
        E.append( h0 + sum( E_ ).real )
    
    fig = plt.figure()
    ax  = fig.add_subplot( 1, 1, 1 )
    ax.plot( vec_k.tick, E )
    ax.set_xticks( vec_k.labeltick )
    ax.set_xticklabels( vec_k.label )
    plt.show()