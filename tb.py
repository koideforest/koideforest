import itertools
import numpy as np
from koideforest import SlaterKoster
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

class Hopping():

    def __init__( self,
                  atoms = ( np.zeros(3), ),
                  orbitals = ( ( 's', ) ),
                  TranslationVector = ((1,0,0),(0,1,0),(0,0,1)) ):

        A        = atoms
        L        = orbitals
        #self.vector = [ [ [] for a2 in A ] for a1 in A ]
        self.vector     = [ [ [ [ [] for l2 in L[ia2] ] for l1 in L[ia1] ] for ia2 in range( len( A ) ) ] for ia1 in range( len( A ) ) ]
        self.TI     = [ [ [ [ [] for l2 in L[ia2] ] for l1 in L[ia1] ] for ia2 in range( len( A ) ) ] for ia1 in range( len( A ) ) ]
        self.T      = TranslationVector
        self.A      = A
        self.L      = L
        
    def add_hop( self, index1, index2, n_set, TI ):
        # index = ( ia, il )

        #self.vector[ index1[0] ][ index2[0] ].append( sum( [ n * tv for n, tv in zip( n_set, self.T ) ] ) )
        self.vector[ index1[0] ][ index2[0] ][ index1[1] ][index2[1]].append( sum( [ n * tv for n, tv in zip( n_set, self.T ) ] ) )
        self.TI[ index1[0] ][ index2[0] ][ index1[1] ][ index2[1] ].append( TI )

        # transpose
        #self.vector[ index2[0] ][ index1[0] ].append( - sum( [ n * tv for n, tv in zip( n_set, self.T ) ] ) )
        self.vector[ index2[0] ][ index1[0] ][ index2[1] ][ index1[1] ].append( - sum( [ n * tv for n, tv in zip( n_set, self.T ) ] ) )
        self.TI[ index2[0] ][ index1[0] ][ index2[1] ][ index1[1] ].append( TI )

    def add_hop_sk( self, index1, index2, n_set, sigma, pi, delta ):
        # index = ( ia, il )
        
        r  = sum( [ n * tv for n, tv in zip( n_set, self.T ) ] )
        ti = SlaterKoster.sk( r, ( sigma, pi, delta ),
                              self.L[ index1[0] ][ index1[1] ], self.L[ index2[0] ][ index2[1] ] )
        
        #self.vector[ index1[0] ][ index2[0] ].append( r )
        self.vector[ index1[0] ][ index2[0] ][ index1[1] ][ index2[1] ].append( r )
        self.TI[ index1[0] ][ index2[0] ][ index1[1] ][ index2[1] ].append( ti )

        # transpose

        ti = SlaterKoster.sk( -r, ( sigma, pi, delta ),
                               self.L[ index2[0] ][ index2[1] ], self.L[ index1[0] ][ index1[1] ] )
        
        #self.vector[ index2[0] ][ index1[0] ].append( r )
        self.vector[ index2[0] ][ index1[0] ][ index2[1] ][ index1[1] ].append( -r )
        self.TI[ index2[0] ][ index1[0] ][ index2[1] ][ index1[1] ].append( ti )

    def Hamiltonian( self, k, ia1, il1, ia2, il2 ):
        return sum([ np.exp( 1j * np.dot( k, r_ ) ) * t_
                     #for r_, t_ in zip( self.vector[ia1][ia2], self.TI[ia1][ia2][il1][il2] ) ])
                     for r_, t_ in zip( self.vector[ia1][ia2][il1][il2], self.TI[ia1][ia2][il1][il2] ) ])


def Neighbor( T, nt, A, dn ):
    # T : basic translation vector
    # nt: number to general translation vector constucted from T
    # A : atmoic site coordination within unit cell
    # dn: distance to nearest neighbor site
    R    = []
    zero = np.zeros( len( T[0] ) ) 

    for a0_ in A:
        R_ = []
        for a1_ in A :
            R__ = []
            # for example ( -1, 0, 1 ) -> ( -1, -1 ), ( -1, 0 ), ( -1, 1 )...
            ijk = itertools.product( np.arange( -nt, nt + 1 ), repeat = len( T ) )
            for i_ in ijk:
                temp = np.array( a1_ ) + sum( [ i_[it] * np.array( t_ ) for it, t_ in enumerate( T ) ] )
                if ( np.linalg.norm( temp - a0_ ) <= dn ) and any( temp - a0_ != zero ):
                    R__.append( temp )
            R_.append( R__ )
        R.append( R_ )
    return R


def Neighbor1_SC( LC ):
    T  = ( np.array( [ LC, 0, 0 ] ), np.array( [ 0, LC, 0 ] ), np.array( [ 0, 0, LC ] ) )
    A  = ( np.zeros(3), )
    dn = LC 
    return Neighbor( T, 1, A, dn )

    #R = []
    #pm = ( -1, 1 )
    #for i in pm:
    #    R.append( a * np.array( [ i, 0, 0 ] ) )
    #    R.append( a * np.array( [ 0, i, 0 ] ) )
    #    R.append( a * np.array( [ 0, 0, i ] ) )
    #return R

def Neighbor2_SC( LC ):
    T  = ( np.array( [ LC, 0, 0 ] ), np.array( [ 0, LC, 0 ] ), np.array( [ 0, 0, LC ] ) )
    A  = ( np.zeros(3), )
    dn = np.sqrt(2) * LC 
    return Neighbor( T, 1, A, dn )
    
    #R = []
    #pm = ( -1, 1 )
    #for i1 in pm:
    #    for i2 in pm:
    #        R.append( a * np.array( [ i1, i2, 0 ] ) )
    #        R.append( a * np.array( [ 0, i1, i2 ] ) )
    #        R.append( a * np.array( [ i2, 0, i1 ] ) )
    #return R

def Neighbor3_SC( LC ):
    T  = ( np.array( [ LC, 0, 0 ] ), np.array( [ 0, LC, 0 ] ), np.array( [ 0, 0, LC ] ) )
    A  = ( np.zeros(3), )
    dn = np.sqrt(3) * LC 
    return Neighbor( T, 1, A, dn )
    
    #R = []
    #pm = ( -1, 1 )
    #for i1 in pm:
    #    for i2 in pm:
    #       for i3 in pm:
    #        R.append( a * np.array( i1, i2, i3 ) )
    #return R

def Neighbor1_BCC( LC ):
    T  = TranslationVector_BCC( LC )
    A  = ( np.zeros(3), )
    dn = np.sqrt(3) * LC/2 
    return Neighbor( T, 1, A, dn )

def Neighbor2_BCC( LC ):
    T  = TranslationVector_BCC( LC )
    A  = ( np.zeros(3), )
    dn = LC
    return Neighbor( T, 1, A, dn )

def Neighbor1_FCC( LC ):
    T  = TranslationVector_FCC( LC )
    A  = ( np.zeros(3), )
    dn = np.sqrt(2)/2 * LC 
    return Neighbor( T, 1, A, dn )

def Neighbor2_FCC( LC ):
    T  = TranslationVector_FCC( LC )
    A  = ( np.zeros(3), )
    dn = LC 
    return Neighbor( T, 1, A, dn )

def Neighbor1_Graphene( LC ):
    # Translational vector
    #t1 = LC * np.array( [  1/2, np.sqrt(3)/2 ] )
    #t2 = LC * np.array( [ -1/2, np.sqrt(3)/2 ] )
    #T  = ( t1, t2 )
    T = TranslationVector_Graphene( LC )

    # atomic site in a unit cell 
    zero = np.zeros(2)
    dn   = LC/np.sqrt(3)  # distance to nearest neighbor site 
    a1   = zero
    a2   = dn * np.array( [ 0, 1 ] )
    A    = ( a1, a2 )

    return Neighbor( T, 1, A, dn )

def TranslationVector_SC( LC ):
    return ( np.array([ LC, 0, 0 ]),
             np.array([ 0, LC, 0 ]), 
             np.array([ 0, 0, LC ])  )

def TranslationVector_BCC( LC ):
    return ( LC/2 * np.array([ -1,  1,  1 ]),
             LC/2 * np.array([  1, -1,  1 ]),
             LC/2 * np.array([  1,  1, -1 ])  )

def TranslationVector_FCC( LC ):
    return ( LC/2 * np.array([ 1, 1, 0 ]),
             LC/2 * np.array([ 0, 1, 1 ]),
             LC/2 * np.array([ 1, 0, 1 ])  )

def TranslationVector_Graphene( LC ):
    t1 = LC * np.array( [  1/2, np.sqrt(3)/2 ] )
    t2 = LC * np.array( [ -1/2, np.sqrt(3)/2 ] )
    return ( t1, t2 )


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

def AtomicCoordination_Graphene( LC ):
    a1   = np.zeros(2)
    a2   = LC/np.sqrt(3) * np.array([ 0, 1 ])
    return ( a1, a2 )
    

def SolveEigen( K, A, T, R, L, E0, V ):
    E, U = [], []
    for k_ in K:
        H = np.zeros( ( np.size( L ), np.size( L ) ), dtype = np.complex )
        for ia1, a1_ in enumerate( A ):
            for ia2, a2_ in enumerate( A ):
                for il1, l1_ in enumerate( L[ia1] ):
                    for il2, l2_ in enumerate( L[ia2] ):
                        i1 = ia1 * len( L[ia1] ) + il1
                        i2 = ia2 * len( L[ia2] ) + il2
                        if i1 == i2:
                            H[i1][i2] += E0[i1]
                        H[i1][i2] += sum( [ np.exp( 1j * np.dot( k_, r_ ) ) * sk( r_, V[i1][i2], l1_, l2_ )
                                            for r_ in R[ia1][ia2] ] )
        value, vector = np.linalg.eigh( H )
        eigen = sorted( [ [ val, vec ] for val, vec in zip( value, vector ) ], key = itemgetter( 0 ) )
        E.append( [ eig[0] for eig in eigen ] )
        U.append( [ eig[1] for eig in eigen ] )
    return np.array( E ).T, np.array( U ).transpose( 1, 2, 0 )

def SolveEigen_Hop( K, E0, Hop ):
    E, U = [], []
    for k_ in K:
        H = np.zeros( ( np.size( Hop.L ), np.size( Hop.L ) ), dtype = np.complex )
        for ia1, a1_ in enumerate( Hop.A ):
            for ia2, a2_ in enumerate( Hop.A ):
                for il1, l1_ in enumerate( Hop.L[ia1] ):
                    for il2, l2_ in enumerate( Hop.L[ia2] ):
                        i1 = ia1 * len( Hop.L[ia1] ) + il1
                        i2 = ia2 * len( Hop.L[ia2] ) + il2
                        if i1 == i2:
                            H[i1][i2] += E0[i1]
                        H[i1][i2] += Hop.Hamiltonian( k_, ia1, il1, ia2, il2 )
        value, vector = np.linalg.eigh( H )
        eigen = sorted( [ [ val, vec ] for val, vec in zip( value, vector ) ], key = itemgetter( 0 ) )
        E.append( [ eig[0] for eig in eigen ] )
        U.append( [ eig[1] for eig in eigen ] )
    return np.array( E ).T, np.array( U ).transpose( 1, 2, 0 )  # U[band][orb][k]

if __name__ == '__main__':

    LC = 2
    R  = Neighbor1_FCC( LC )

    vec_k = ReciprocalVector_band( )
    vec_k.initial_k_point( list( np.pi/LC * np.array( [ 1, 1, 1 ] ) ), 'L' )
    vec_k.add_k_point( [ 0 ] * 3, '$\Gamma$' ) 
    vec_k.add_k_point( [ 2*np.pi/LC, 0, 0  ], 'X' )
 
    # s band (Slater-Koster)
    h0    = -1
    sigma = -0.2
    T     = SlaterKoster.ss( R, sigma )
    E     = []
    for k_ in vec_k.k:
        E_ = [ np. exp( 1j * np.dot( k_, r_ ) ) * SlaterKoster.ss( r_, sigma ) for r_ in R[0][0] ]
        E.append( h0 + sum( E_ ).real )
    
    fig = plt.figure()
    ax  = fig.add_subplot( 1, 1, 1 )
    ax.plot( vec_k.axis, E )
    ax.set_xticks( vec_k.tick )
    ax.set_xticklabels( vec_k.ticklabel )
    plt.show()

    # s band (Hopping integral)

    T = TranslationVector_FCC( LC )
    
    A  = ( np.zeros(3), )
    L  = ( ( 's', ) )
    E0      = np.zeros( np.size( L ) )
    E0[ 0 ] = -1.0
    sigma   = -0.2

    Hop = Hopping( A, L, T )
    Hop.add_hop( ( 0, 0 ), ( 0, 0 ), (  1,  0,  0 ), sigma )
    Hop.add_hop( ( 0, 0 ), ( 0, 0 ), (  0,  1,  0 ), sigma )
    Hop.add_hop( ( 0, 0 ), ( 0, 0 ), (  0,  0,  1 ), sigma )
    Hop.add_hop( ( 0, 0 ), ( 0, 0 ), (  0, -1,  1 ), sigma )
    Hop.add_hop( ( 0, 0 ), ( 0, 0 ), (  1,  0, -1 ), sigma )
    Hop.add_hop( ( 0, 0 ), ( 0, 0 ), ( -1,  1,  0 ), sigma )

    E, U = SolveEigen_Hop( vec_k.k, E0, Hop )
    
    fig = plt.figure()
    ax  = fig.add_subplot( 1, 1, 1 )
    ax.plot( vec_k.axis, E[0] )
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
                                        * SlaterKoster.pp( r_, sigma, pi, xyz[i1], xyz[i2] ) for r_ in R[0][0] ] )
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

