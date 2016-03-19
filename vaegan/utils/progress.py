#coding : utf-8

import sys

class Progress( object ):
    def __init__( self, max_count, size ):
        if size <= max_count:
            self.__size = size
        else:
            self.__size = max_count

        self.__max_count = max_count
        self.__sep = int(max_count/size) + 1
        self.__count = 0

    def prog( self ):
        if int( self.__count % self.__sep ) != 0:
            self.__count += 1
            return
        p = int( self.__count / self.__sep ) + 1
        s = u'|' + u'=' * p + u' ' * (self.__size-p) + u'| %d/%d' \
            % (self.__count,self.__max_count)
        sys.stdout.write("\r%s" % s)
        sys.stdout.flush()

        self.__count += 1

    def end( self ):
        self.__count = 0
        p = self.__size
        s = u'|' + u'=' * p + u' ' * (self.__size-p) + u'| %d/%d' \
            % (self.__max_count ,self.__max_count)
        sys.stdout.write("\r%s" % s)
        sys.stdout.flush()
        print >>sys.stdout

if __name__ == '__main__':
    a = range(1000000)
    prog = Progress(len(a),50)

    for e in range(10):
        for i,v in enumerate(a):
            prog.prog()
        prog.end()
