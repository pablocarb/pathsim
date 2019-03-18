'''
sampleCompression (c) University of Manchester 2019

sampleCompression is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  Pablo Carbonell
@description: Evaluate design efficiency in function of sample space . 
'''

from doebase.doebase import doeTemplate, promoterList, plasmidList, read_excel
from doebase import OptDes
import os
import numpy as np
from itertools import product
import csv




def evaldes( steps, variants, npromoters, nplasmids, libsize, positional, 
             outfile=None, random=False ):

    plasmids = plasmidList(nplasmids)
    promoters = promoterList(npromoters)
    
    tree = []
    genes = {}
    for i in np.arange(steps):
        rid = "r%0d" % (i,)
        tree.append(rid)
        genes[rid] = []
        for j in np.arange(variants):
            gid = "g%0d_%0d" % (i,j)
            genes[rid].append(gid)
            
    doe = doeTemplate( tree, plasmids, promoters, genes, positional )
    if outfile is not None:
        doe.to_excel( outfile, index=False )
        fact, partinfo = read_excel( outfile )
    else:
        fact, partinfo = read_excel( None, doedf=doe )
    try:
        seed = np.random.randint(10000)
        starts = 1
        RMSE = 10
        alpha = 0.05
        factors, fnames, diagnostics = OptDes.makeDoeOptDes(fact, size=libsize, 
                                                            seed=seed, starts=starts,
                                                            RMSE= RMSE, alpha=alpha,
                                                            random=random )
    except:
        raise Exception("No solution")
    diagnostics['steps'] = steps
    diagnostics['variants'] = variants
    diagnostics['npromoters'] = npromoters
    diagnostics['nplasmids'] = nplasmids
    diagnostics['libsize'] = libsize
    return diagnostics

if __name__ == '__main__':
    out = os.path.join(os.getenv('DATA'),'doecomp')
    
    TEST = 3   
        
    if TEST == 1:
        """ Initial test: try multiple combinations.
            The issue is that numbers are almost all the same
            because some params compensate the others
        """
        rsteps = [4,6,8,10]
        rvariants = [1,5,10]
        rpromoters = [1,3,5]
        rplasmids = [1,2]
        rpositional = [False,True]
        outres = os.path.join(out,'res.csv')
        def nextLib(libsize):
            return libsize * 2
        def variations(var):
            return product(*var)
    elif TEST == 2:
        """ This new test will be more focused, less variations (only 2 params?)
            but a finer step in the libsize. Probably is better keeping
            exponential increase in the library size """
        rsteps = [4,6,8,10]
        rvariants = np.arange(1,10)
        rpromoters = [3]
        rplasmids = [2]
        rpositional = [False]
        outres = os.path.join(out,'res2test.csv')
        def nextLib(libsize):
            return libsize * 1.2
        def variations(var):
            return product(*var)
    elif TEST == 3:
        """ Random test
        """
        rsteps = [4,6,8,10]
        rvariants = [1,5,10]
        rpromoters = [1,3,5]
        rplasmids = [1,2]
        rpositional = [False]
        outres = os.path.join(out,'res7.csv')
        def nextLib(libsize):
            return np.random.randint(128) 
        def variations(var):
            rows = []
            for j in np.arange(0,1e3):
                x = []
                for v in var:
                    x.append( np.random.choice(v) )
                x[-1] = bool(x[-1])
                rows.append( x )
            return rows
    
    var = [ rsteps, rvariants, rpromoters, rplasmids, rpositional ]
    with open(outres, 'w') as h:
        cw = csv.writer(h)
        cw.writerow( ('steps', 'variants', 'npromoters', 'nplasmids', 'pos', 'libsize', 'eff', 'space', 'pow', 'rpv') )
        for steps, variants, npromoters, nplasmids, positional in variations( var ):
            J = 0
            n = 1
            libsize = 8
            # Try the same design multiple times with several libraries
            while J<98 and n<10: 
                row = (steps, variants, npromoters, nplasmids)
                n += 1
                libsize = nextLib( libsize )
                print(row, libsize, n)
                try:
                    factors, df, diagnostics = evaldes( steps, variants, npromoters, nplasmids, libsize, positional )
                    J = diagnostics['J']
                    pows = diagnostics['pow']
                    rpvs = diagnostics['rpv']
                    v = [len(x) for x in factors]
                    if positional:
                        pos = 1
                    else:
                        pos = 0
                except:
                    continue
                try:
                    pown = np.mean(pows)
                except:
                    pown = 0
                try:
                    rpvn = np.mean(rpvs)
                except:
                    rpvn = np.nan
                row = (steps, variants, npromoters, nplasmids, pos, libsize, J, np.prod(v), pown, rpvn)
                print( row )
                cw.writerow( row )
                h.flush()
