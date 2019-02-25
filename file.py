import numpy as np
from copy import copy
from matplotlib import pyplot as plt

def StripElementInLine( l, back_up = False ):
    if back_up:
        ll = copy( l )
    else:
        ll = l
    for i, e in enumerate( ll ):
        if type( e ) == str:
            ll[ i ] = e.strip()
    return ll

def FloatList( l, back_up = False ):
    if back_up:
        l_ = copy( l )
    else:
        l_ = l
    l_ = list( map( float, l_ ) )
    return l_

def FloatData( data, back_up = False ):
    if back_up:
        data_ = copy( data )
    else:
        data_ = data
    for i, d in enumerate( data_ ):
        data_[ i ] = list( map( float, d ) )
    return data_

def Line2List( line, separator = ' ' ):
    l = line.split( separator )
    while '' in l:
        l.remove( '' )
    return l

def ReadData( file_path, separator = ' ' ):
    f = open( file_path, 'r' )
    lines = f.read().split( '\n' )
    f.close()
    data = []
    for i, line in enumerate( lines ):
        if i == len( lines ) - 1:
            if not line:
                break
        l = Line2List( line, separator )
        data.append( Line2List( line, separator ) )
    return data

def WriteData( file_path, label, x, y, format_x, format_y ):
    f = open( file_path, 'w' )
    f.write( label + '\n' )
    for x_, y_ in zip( x, y ):
        f.write( format_x.format( x_ ) )
        if not hasattr( y_, '__iter__' ):
            f.write( format_y.format( y_ ) )
        else:
            for e in y_:
                f.write( format_y.format( e ) )
        f.write( '\n' )
    f.close()

def PlotData( data_xy ):
    x = data_xy[ 0 ]
    for y in data_xy[ 1: ]:
        plt.plot( x, y )
    plt.show()


class Data():

    def __init__( self, file_path, separator = ' ' ):
        self.data_org = ReadData( file_path, separator )
        self.data     = ReadData( file_path, separator )

    # For example, removing labels
    def RemoveLines( self, line_list = [ 0, ] ):
        for ll in reversed( line_list ):
            self.data.pop( ll ) 

    def Float( self ):
        for i, d in enumerate( self.data ):
            self.data[ i ] = list( map( float, d ) )

    def XYStyle( self ):
        self.data = np.array( self.data ).T

    def QuickPlot( self ):
        data_xy = np.array( FloatData( self.data_org, back_up = True ) ).T
        PlotData( data_xy )

    def Plot( self ):
        # self.RemoveLabel( label_line )
        # self.Float()
        # self.XYStyle()
        PlotData( self.data )
