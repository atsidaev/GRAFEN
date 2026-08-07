# -*- coding: utf-8 -*-

import numpy as np
from math import fabs, floor
from struct import Struct, pack

class GridExc_base(Exception): pass
class GridExc_UnexpectedFileFormat(GridExc_base): pass
class GridExc_AttributeError(GridExc_base): pass

class Grid(object):
    BlankValue = 1.70141e+038
    parser_header = Struct("<ii")
    parser_GridSection = Struct("<ii8d")
    parser_4write = Struct("<7i8dii")
    BlankValueBin = pack("<d", BlankValue)
    Eps4FloatCompare = 1.0e-10
    
    def __init__(self, xnum=2, ynum=2, xmin=0.0, ymin=0.0, dx=1.0, dy=1.0, zmin=0.0, zmax=0.0, BlankValue=1.70141e+038):
        self.ynum = ynum
        self.xnum = xnum
        self.xmin = xmin
        self.ymin = ymin
        self.dx = dx
        self.dy = dy
        self.zmin = zmin
        self.zmax = zmax
        self.BlankValue = BlankValue
        self.data = None

    def copy(self):
        g = Grid(self.xnum, self.ynum, self.xmin, self.ymin, self.dx, self.dy, self.zmin, self.zmax, self.BlankValue)
        if self.data is not None: g.data = self.data.copy()
        return g

    def copy_header(self):
        return Grid(self.xnum, self.ynum, self.xmin, self.ymin, self.dx, self.dy, self.zmin, self.zmax, self.BlankValue)

    def datainit(self):
        if self.data is None: self.data = np.empty(self.size)
        return self

    @property
    def xmax(self): return self.xmin + self.dx*float(self.xnum-1)

    @property
    def ymax(self): return self.ymin + self.dy*float(self.ynum-1)

    @property
    def size(self): return self.xnum*self.ynum

    def __getitem__(self, key):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(key, int):
            if key < 0 or key >= self.size: raise IndexError()
            ix = key%self.xnum
            iy = key/self.xnum
            return (self.xmin+self.dx*float(ix), self.ymin+self.dy*float(iy), self.data[key])
        elif isinstance(key, tuple) and len(key) > 1 and isinstance(key[0], int) and isinstance(key[1], int):
            if key[0] < 0 or key[0] >= self.xnum or key[1] < 0 or key[1] >= self.ynum: raise IndexError()
            return (self.xmin+self.dx*float(key[0]), self.ymin+self.dy*float(key[1]), self.data[self.xnum*key[1]+key[0]])
        else: raise TypeError("type(key) must be int or tuple (int, int)")

    def __setitem__(self, key, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(key, int):
            if key < 0 or key >= self.size: raise IndexError()
            self.data[key] = float(val)
        elif isinstance(key, tuple) and len(key) > 1 and isinstance(key[0], int) and isinstance(key[1], int):
            if key[0] < 0 or key[0] >= self.xnum or key[1] < 0 or key[1] >= self.ynum: raise IndexError()
            self.data[self.xnum*key[1]+key[0]] = float(val)
        else: raise TypeError("type(key) must be int or tuple (int, int)")

    def __add__(self, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(val, Grid):
            if val.data is None: raise GridExc_AttributeError("val.data is None")
            if not self.compare_headers(val): raise GridExc_AttributeError("grids have different headers")
            g = self.copy()
            for i, v1 in enumerate(g.data):
                v2 = val.data[i]
                if v1 >= g.BlankValue or v2 >= val.BlankValue: g.data[i] = g.BlankValue
                else: g.data[i] = v1 + v2
            return g
        elif isinstance(val, float) or isinstance(val, int):
            g = self.copy()
            for i, v1 in enumerate(g.data):
                if v1 < g.BlankValue: g.data[i] = v1 + val
            return g
        else: return NotImplemented

    def __sub__(self, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(val, Grid):
            if val.data is None: raise GridExc_AttributeError("val.data is None")
            if not self.compare_headers(val): raise GridExc_AttributeError("grids have different headers")
            g = self.copy()
            for i, v1 in enumerate(g.data):
                v2 = val.data[i]
                if v1 >= g.BlankValue or v2 >= val.BlankValue: g.data[i] = g.BlankValue
                else: g.data[i] = v1 - v2
            return g
        elif isinstance(val, float) or isinstance(val, int):
            g = self.copy()
            for i, v1 in enumerate(g.data):
                if v1 < g.BlankValue: g.data[i] = v1 - val
            return g
        else: return NotImplemented

    def __mul__(self, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(val, Grid):
            if val.data is None: raise GridExc_AttributeError("val.data is None")
            if not self.compare_headers(val): raise GridExc_AttributeError("grids have different headers")
            g = self.copy()
            for i, v1 in enumerate(g.data):
                v2 = val.data[i]
                if v1 >= g.BlankValue or v2 >= val.BlankValue: g.data[i] = g.BlankValue
                else: g.data[i] = v1 * v2
            return g
        elif isinstance(val, float) or isinstance(val, int):
            g = self.copy()
            for i, v1 in enumerate(g.data):
                if v1 < g.BlankValue: g.data[i] = v1 * val
            return g
        else: return NotImplemented

    def __truediv__(self, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(val, Grid):
            if val.data is None: raise GridExc_AttributeError("val.data is None")
            if not self.compare_headers(val): raise GridExc_AttributeError("grids have different headers")
            g = self.copy()
            for i, v1 in enumerate(g.data):
                v2 = val.data[i]
                if v1 >= g.BlankValue or v2 >= val.BlankValue or v2 == 0.0: g.data[i] = g.BlankValue
                else: g.data[i] = v1 / v2
            return g
        elif isinstance(val, float) or isinstance(val, int):
            g = self.copy()
            if val == 0:
                g.data.fill(g.BlankValue)
                return g
            val1 = 1.0/float(val)
            for i, v1 in enumerate(g.data):
                if v1 < g.BlankValue: g.data[i] = v1 * val1
            return g
        else: return NotImplemented

    def __iadd__(self, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(val, Grid):
            if val.data is None: raise GridExc_AttributeError("val.data is None")
            if not self.compare_headers(val): raise GridExc_AttributeError("grids have different headers")
            for i, v1 in enumerate(self.data):
                v2 = val.data[i]
                if v1 >= self.BlankValue or v2 >= val.BlankValue: self.data[i] = self.BlankValue
                else: self.data[i] = v1 + v2
            return self
        elif isinstance(val, float) or isinstance(val, int):
            for i, v1 in enumerate(self.data):
                if v1 < self.BlankValue: self.data[i] = v1 + val
            return self
        else: return NotImplemented

    def __isub__(self, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(val, Grid):
            if val.data is None: raise GridExc_AttributeError("val.data is None")
            if not self.compare_headers(val): raise GridExc_AttributeError("grids have different headers")
            for i, v1 in enumerate(self.data):
                v2 = val.data[i]
                if v1 >= self.BlankValue or v2 >= val.BlankValue: self.data[i] = self.BlankValue
                else: self.data[i] = v1 - v2
            return self
        elif isinstance(val, float) or isinstance(val, int):
            for i, v1 in enumerate(self.data):
                if v1 < self.BlankValue: self.data[i] = v1 - val
            return self
        else: return NotImplemented

    def __imul__(self, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(val, Grid):
            if val.data is None: raise GridExc_AttributeError("val.data is None")
            if not self.compare_headers(val): raise GridExc_AttributeError("grids have different headers")
            for i, v1 in enumerate(self.data):
                v2 = val.data[i]
                if v1 >= self.BlankValue or v2 >= val.BlankValue: self.data[i] = self.BlankValue
                else: self.data[i] = v1 * v2
            return self
        elif isinstance(val, float) or isinstance(val, int):
            for i, v1 in enumerate(self.data):
                if v1 < self.BlankValue: self.data[i] = v1 * val
            return self
        else: return NotImplemented

    def __itruediv__(self, val):
        if self.data is None: raise GridExc_base("self.data is None")
        if isinstance(val, Grid):
            if val.data is None: raise GridExc_AttributeError("val.data is None")
            if not self.compare_headers(val): raise GridExc_AttributeError("grids have different headers")
            for i, v1 in enumerate(self.data):
                v2 = val.data[i]
                if v1 >= self.BlankValue or v2 >= val.BlankValue or v2 == 0.0: self.data[i] = self.BlankValue
                else: self.data[i] = v1 / v2
            return self
        elif isinstance(val, float) or isinstance(val, int):
            if val == 0:
                self.data.fill(self.BlankValue)
                return self
            val1 = 1.0/float(val)
            for i, v1 in enumerate(self.data):
                if v1 < self.BlankValue: self.data[i] = v1 * val1
            return self
        else: return NotImplemented

    def __pos__(self):
        if self.data is None: raise GridExc_base("self.data is None")
        return self.copy()

    def __neg__(self):
        if self.data is None: raise GridExc_base("self.data is None")
        g = self.copy()
        for i, v1 in enumerate(g.data):
            if v1 < self.BlankValue: g.data[i] = -v1
        return g

    def mean(self):
        if self.data is None: return None
        s = 0.0
        n = 0
        for v in self.data:
            if v < self.BlankValue:
                s += v
                n += 1
        if n != 0: return s/float(n)
        else: return None

    def read_grd7(self, filename: str):
        with open(filename, "rb") as f:
            fend = f.seek(0,2)
            f.seek(0,0)
            id, size = Grid.parser_header.unpack(f.read(Grid.parser_header.size))
            if id != 0x42525344 or f.seek(size, 1) >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))

            id, size = Grid.parser_header.unpack(f.read(Grid.parser_header.size))
            while id != 0x44495247:
                if f.seek(size, 1) >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))
                id, size = Grid.parser_header.unpack(f.read(Grid.parser_header.size))
            if size != Grid.parser_GridSection.size or f.tell() >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))

            self.ynum, self.xnum, self.xmin, self.ymin, self.dx, self.dy, self.zmin, self.zmax, _, self.BlankValue = Grid.parser_GridSection.unpack(f.read(Grid.parser_GridSection.size))
            if f.tell() >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))

            id, size = Grid.parser_header.unpack(f.read(Grid.parser_header.size))
            if id != 0x41544144 or size != self.xnum*self.ynum*8: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))
            if f.tell() >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))

            self.data = np.fromfile(f, dtype=float, count=self.xnum*self.ynum)
        #self.data.shape = (self.ynum, self.xnum)
        return self
    
    def read_grd7_header(self, filename: str):
        with open(filename, "rb") as f:
            fend = f.seek(0,2)
            f.seek(0,0)
            id, size = Grid.parser_header.unpack(f.read(Grid.parser_header.size))
            if id != 0x42525344 or f.seek(size, 1) >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))

            id, size = Grid.parser_header.unpack(f.read(Grid.parser_header.size))
            while id != 0x44495247:
                if f.seek(size, 1) >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))
                id, size = Grid.parser_header.unpack(f.read(Grid.parser_header.size))
            if size != Grid.parser_GridSection.size or f.tell() >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))

            self.ynum, self.xnum, self.xmin, self.ymin, self.dx, self.dy, self.zmin, self.zmax, _, self.BlankValue = Grid.parser_GridSection.unpack(f.read(Grid.parser_GridSection.size))
            if f.tell() >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))

            id, size = Grid.parser_header.unpack(f.read(Grid.parser_header.size))
            if id != 0x41544144 or size != self.xnum*self.ynum*8: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))
            if f.tell() >= fend: raise GridExc_UnexpectedFileFormat("\"{}\" is not a grd7 file".format(filename))

        self.data = None
        return self

    def write_grd7(self, filename: str):
        with open(filename, "wb") as f:
            f.write(Grid.parser_4write.pack(0x42525344, 4, 1, 0x44495247, Grid.parser_GridSection.size, self.ynum, self.xnum, self.xmin, self.ymin, self.dx, self.dy, self.zmin, self.zmax, 0.0, self.BlankValue, 0x41544144, 8*self.size))
            if self.data is not None: self.data.tofile(f)
            else:
                for _ in range(self.size): f.write(Grid.BlankValueBin)
        return self

    def write_xyz(self, filename: str, writeBlanks = False, writeHeader = True):
        with open(filename, "wt") as f:
            if writeBlanks:
                if writeHeader: f.write("{0:d}\n".format(self.size))
                y = self.ymin
                if self.data is None:
                    for iy in range(self.ynum):
                        x = self.xmin
                        for ix in range(self.xnum):
                            f.write("{0:.12g} {1:.12g} {2:.12g}\n".format(x, y, self.BlankValue))
                            x += self.dx
                        y += self.dy
                else:
                    n = 0
                    for iy in range(self.ynum):
                        x = self.xmin
                        for ix in range(self.xnum):
                            f.write("{0:.12g} {1:.12g} {2:.12g}\n".format(x, y, self.data[n]))
                            n += 1
                            x += self.dx
                        y += self.dy
            elif self.data is not None:
                if writeHeader: f.write("          \n") #10 пробелов
                y = self.ymin
                n = 0
                counter = 0
                for iy in range(self.ynum):
                    x = self.xmin
                    for ix in range(self.xnum):
                        if self.data[n] < self.BlankValue:
                            f.write("{0:.12g} {1:.12g} {2:.12g}\n".format(x, y, self.data[n]))
                            counter += 1
                        n += 1
                        x += self.dx
                    y += self.dy
                if writeHeader:
                    f.seek(0)
                    f.write("{0:d}".format(counter))
        return self

    def fixzminmax(self):
        if self.data is not None:
            self.zmin = np.amin(self.data)
            self.zmax = np.amax(self.data)
        return self

    def enumerate1d(self):
        y = self.ymin
        n = 0
        if self.data is None:
            for iy in range(self.ynum):
                x = self.xmin
                for ix in range(self.xnum):
                    yield (n, x, y, None)
                    x += self.dx
                    n += 1
                y += self.dy
        else:
            for iy in range(self.ynum):
                x = self.xmin
                for ix in range(self.xnum):
                    yield (n, x, y, self.data[n])
                    x += self.dx
                    n += 1
                y += self.dy
    
    def enumerate2d(self):
        y = self.ymin
        if self.data is None:
            for iy in range(self.ynum):
                x = self.xmin
                for ix in range(self.xnum):
                    yield (ix, iy, x, y, None)
                    x += self.dx
                y += self.dy
        else:
            n = 0
            for iy in range(self.ynum):
                x = self.xmin
                for ix in range(self.xnum):
                    yield (ix, iy, x, y, self.data[n])
                    n += 1
                    x += self.dx
                y += self.dy

    def compare_headers(self, grd):
        if not isinstance(grd, Grid): raise GridExc_AttributeError("grd is not instance of pygrid.Grid")
        return self.xnum == grd.xnum and self.ynum == grd.ynum and fabs(self.xmin - grd.xmin) < Grid.Eps4FloatCompare and fabs(self.ymin - grd.ymin) < Grid.Eps4FloatCompare and fabs(self.dx - grd.dx) < Grid.Eps4FloatCompare and fabs(self.dy - grd.dy) < Grid.Eps4FloatCompare

    def funcfill_coord(self, func):
        "float func(x:float, y:float)"
        if self.data is None: self.data = np.empty(self.size)
        n = 0
        y = self.ymin
        for iy in range(self.ynum):
            x = self.xmin
            for ix in range(self.xnum):
                self.data[n] = func(x, y)
                n += 1
                x += self.dx
            y += self.dy
        return self

    def map_inpl_z2z(self, func, what_to_do_with_blank = 1, defaultZ = 0.0):
        """float func(z:float).
        if what_to_do_with_blank >0: бланки оставлять как есть (НЕ передавать их в f).
        if what_to_do_with_blank==0: вместо бланков передавать в f значение z.
        if what_to_do_with_blank <0: в f передается все без проверки (что делать с бланками - решает f)."""
        
        if self.data is None: return self
        if what_to_do_with_blank > 0:
            for n,val in enumerate(self.data):
                if self.data[n] < self.BlankValue: self.data[n] = func(val)
        elif what_to_do_with_blank == 0:
            for n,val in enumerate(self.data):
                if self.data[n] < self.BlankValue: self.data[n] = func(val)
                else: self.data[n] = func(defaultZ)
        else:
            for n,val in enumerate(self.data):
                self.data[n] = func(val)
        return self

    def funcfill_index(self, func):
        "float func(n:int)"
        if self.data is None: self.data = np.empty(self.size)
        for n in range(self.size): self.data[n] = func(n)
        return self

    def funcfill_index2(self, func):
        "float func(ix:int, iy:int)"
        if self.data is None: self.data = np.empty(self.size)
        n = 0
        for iy in range(self.ynum):
            for ix in range(self.xnum):
                self.data[n] = func(ix, iy)
                n += 1
        return self

    def getindex2(self, x:float, y:float):
        return (floor((x-self.xmin)/self.dx+0.5), floor((y-self.ymin)/self.dy+0.5))

    def getindex(self, x:float, y:float):
        ix,iy = (floor((x-self.xmin)/self.dx+0.5), floor((y-self.ymin)/self.dy+0.5))
        #ix,iy = self.getindex2(x,y)
        if 0<=ix<self.xnum and 0<=iy<self.ynum:
            return ix+self.xnum*iy
        else: return -1

    def getvalue(self, x:float, y:float, defval=None):
        ix,iy = (floor((x-self.xmin)/self.dx+0.5), floor((y-self.ymin)/self.dy+0.5))
        #ix,iy = self.getindex2(x,y)
        if 0<=ix<self.xnum and 0<=iy<self.ynum:
            return self.data[ix+self.xnum*iy]
        elif defval is None: return self.BlankValue
        else: return defval
    
    def blank2nan(self):
        for i,v in enumerate(self.data):
            if v >= self.BlankValue:
                self.data[i] = np.nan
        return self

    def blank2val(self, value):
        for i,v in enumerate(self.data):
            if v >= self.BlankValue:
                self.data[i] = value
        return self

    def addcells(self, left_add, right_add, bottom_add, top_add, defval = BlankValue):
        """Добавляет (убирает, если параметр отрицательный)
        left_add, right_add, bottom_add, top_add ячеек к гриду."""
        xmin_new = self.xmin - self.dx*left_add
        ymin_new = self.ymin - self.dy*bottom_add
        xnum_new = self.xnum + left_add + right_add
        ynum_new = self.ynum + bottom_add + top_add
        if xnum_new < 2:
            if ynum_new < 2:
                return Grid(2, 2, xmin_new, ymin_new, self.dx, self.dy, 0.0, 0.0, self.BlankValue)
            else:
                return Grid(2, ynum_new, xmin_new, ymin_new, self.dx, self.dy, 0.0, 0.0, self.BlankValue)
        elif ynum_new < 2:
                return Grid(xnum_new, 2, xmin_new, ymin_new, self.dx, self.dy, 0.0, 0.0, self.BlankValue)
        g = Grid(xnum_new, ynum_new, xmin_new, ymin_new, self.dx, self.dy, 0.0, 0.0, self.BlankValue)
        if self.data is None:
            return g
        g.data = np.full(g.size, float(defval))

        ixns = left_add
        if ixns < 0: ixns = 0
        elif ixns >= g.xnum: return g
        ixne = self.xnum + left_add
        if ixne <= 0: return g
        elif ixne > g.xnum: ixne = g.xnum
        iyns = bottom_add
        if iyns < 0: iyns = 0
        elif iyns >= g.ynum: return g
        iyne = self.ynum + bottom_add
        if iyne <= 0: return g
        elif iyne > g.ynum: iyne = g.ynum
        ixos = ixns - left_add
        iyos = iyns - bottom_add
        #ixoe = ixne - left_add
        #iyoe = iyne - bottom_add

        datan = g.data
        datao = self.data
        ixnr = ixne - ixns
        iynr = iyne - iyns
        nn_dy = g.xnum - ixnr
        #no_dy = self.xnum + ixos - ixoe
        no_dy = self.xnum - ixnr
        nn = ixns + iyns*g.xnum
        no = ixos + iyos*self.xnum
        #for iyn in range(iyns, iyne):
        for _ in range(iynr):
            #for ixn in range(ixns, ixne):
            for _ in range(ixnr):
                datan[nn] = datao[no]
                nn += 1
                no += 1
            nn += nn_dy
            no += no_dy

        return g

def read_grd7(filename: str):
    return Grid().read_grd7(filename)

def read_grd7_header(filename: str):
    return Grid().read_grd7_header(filename)

def create(xmin:float, xmax:float, xnum:int, ymin:float, ymax:float, ynum:int, BlankValue=1.70141e+038):
    xnum2 = xnum
    if xnum < 2: xnum2 = 2
    ynum2 = ynum
    if ynum < 2: ynum2 = 2
    return Grid(xnum=xnum2, ynum=ynum2, xmin=xmin, ymin=ymin, dx=(xmax-xmin)/float(xnum2-1), dy=(ymax-ymin)/float(ynum2-1), zmin=0.0, zmax=0.0, BlankValue=BlankValue)

__all__ = ["Grid", "read_grd7", "read_grd7_header", "create", "GridExc_base", "GridExc_UnexpectedFileFormat", "GridExc_AttributeError"]

if __name__ == "__main__":
    g = Grid(2, 3, -1.0, 1.0, 1.0, 1.0)
    g.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    g.fixzminmax()
    g.write_grd7("test.grd")
    g2 = Grid().read_grd7("test.grd")
    #g2.read_grd7("test.grd")
    print(g2.data)
    x = np.zeros(g.size)
    y = np.zeros(g.size)
    for n,x[n],y[n],_ in g.enumerate1d(): pass
    print(x)
    print(y)
    g2 -= g2.mean()
    print(g2.data)

