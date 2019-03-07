# -*- coding: utf-8 -*-

'''
pathSim (c) University of Manchester 2019

pathSim is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  Pablo Carbonell
@description: Basic pathway simulation
'''

import tellurium as te
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from itertools import product
import re
import matplotlib.pyplot as plt
from sampleCompression import evaldes



def modelHeader():
    antinom = """
    // Created by libAntimony v2.9.4
        function Constant_flux__irreversible(v)
          v;
        end

        function Henri_Michaelis_Menten__irreversible(substrate, enzyme, Km, kcat)
          kcat*enzyme*substrate/(Km + substrate);
        end

        function Hill_Cooperativity(substrate, Shalve, V, h)
          V*(substrate/Shalve)^h/(1 + (substrate/Shalve)^h);
        end
    """
    return antinom

def modelTemplate(promoter):
    """ Nsteps basic linear pathway defined using tellurium """
    antinom = ''
    if promoter is not None:
        antinom += """
            model Prom_Model()
        """
    else:
        antinom += """
            model Noprom_Model()
        """
    antinom += """
          // Compartments and Species:
          compartment Cell;
          
          species Substrate in Cell, Product in Cell, Enzyme in Cell;
          """
    if promoter is not None:
        antinom += """
              species Inducer in Cell; 
          """
    antinom += """
          species Activated_promoter in Cell;
          species Growth in Cell;
          Biomass: Growth -> Substrate; Cell*Kg1*Growth - Cell*Kg2*Substrate
           Decay: Growth -> ; Cell*Kd*Growth
          // Reactions:
          //Induc: => Inducer; Cell*Constant_flux__irreversible(1);
          // See doi: https://doi.org/10.1101/360040 for modeling the induction using the Hill function
          """
    if promoter is not None:
        antinom += """
          Induction: Inducer => Activated_promoter; Copy_number*Cell*Hill_Cooperativity(Inducer, Induction_Shalve, Induction_Vi, Induction_h);
          """
    antinom += """
          Expression: Activated_promoter => Enzyme; Copy_number*Cell*Expression_k1*Activated_promoter;
          Leakage:  => Enzyme; Cell*Constant_flux__irreversible(Leakage_vl);
          Degradation: Enzyme => ; Cell*Degradation_k2*Enzyme;
          Catalysis: Substrate => Product; Cell*Henri_Michaelis_Menten__irreversible(Substrate, Enzyme, Catalysis_Km, Catalysis_kcat);

         // Species initializations:
          Substrate = 0.5;
          Product = 0;
          Enzyme = 0;
          """
    if promoter is not None:
        antinom += """
          Inducer = 1;
         """
    antinom += """
          Activated_promoter = 0;
          Copy_number = 1;

          // Compartment initializations:
          Cell = 1;
          Growth = 1;

          // Variable initializations:
          Induction_Shalve = 1e-1;
          Induction_Vi = 1e7;
          Induction_h = 1.85;
          Expression_k1 = 1e6;
          Leakage_vl = 1e-9;
          Degradation_k2 = 1e-6;
          Catalysis_Km = 0.1;
          Catalysis_kcat = 0.1;
          Kg1 = 5;
          Kg2 = 1;
          Kd = 1e-1;
          

          // Other declarations:
          const Cell;

        end
        
        """
    return antinom
        
        
def pathway(promoters):         
    antinom = modelHeader()
    antinom += modelTemplate(1)
    antinom += modelTemplate(None)    
    antinom += "model *Big_Model()"+"\n"
    for i in np.arange(len(promoters)):
        p = promoters[i]
        if p is not None:
            antinom += "\t"+"m%d: Prom_Model();" % (i+1,)
        else:
            antinom += "\t"+"m%d: Noprom_Model();" % (i+1,)
        antinom += "\n"
    for i in np.arange(len(promoters)-1):
        antinom += "\t"+"m%d.Product is m%d.Substrate;" % (i+1, i+2)
        antinom += "\n"
    for i in np.arange(1,len(promoters)):
        p = promoters[i]
        if p is None:
            antinom += "\t"+"m%d.Activated_promoter is m%d.Activated_promoter" %(i+1,i)
            antinom += "\n"
    antinom += "end\n"
    return te.loada(antinom)


class Model():
    def __init__(self, nsteps, promoters):
        self.nsteps = nsteps
        self.promoters = promoters
        self.model = pathway(promoters)
        self.kinetics = None
        self.copy_number = None
        self.leakage = None
        self.degradation = None
        self.SetPromoters()
    def SetKinetics(self,kinetics):
        self.kinetics = kinetics
        for i in np.arange(kinetics):
            self.model['m'+str(i+1)+'_Catalysis_kcat'] = kinetics[i][0] 
            self.model['m'+str(i+1)+'_Catalysis_Km'] = kinetics[i][1] 
    def SetCopyNumber(self,cn):
        self.copy_number = cn
        for i in np.arange(self.nsteps):
            self.model['m'+str(i+1)+'_Copy_number'] = cn 
    def SetPromoters(self):
        for i in np.arange(self.nsteps):
            if self.promoters[i] is not None:
                self.model['m'+str(i+1)+'_Expression_k1'] = self.promoters[i]
            else:
                self.model['m'+str(i+1)+'_Expression_k1'] = self.model['m'+str(i)+'_Expression_k1']
    def SetLeakage(self,leaks):
        for i in np.arange(self.nsteps):
             self.model['m'+str(i+1)+'_Leakage_vl'] = leaks[i]
    def SetDegradation(self,deg):
        for i in np.arange(self.nsteps):
             self.model['m'+str(i+1)+'_Degradation_k2'] = deg[i]
                
    
            

def ranges():
    """ Define global ranges for random parameters """
    param = {
        'Catalysis': {
                'Km': [0.1, 100],
                'kcat': [0.1, 10] 
                },
        'Degradation': {
                'k2': [1e-6, 1e-6]
                },
        'Induction': {
                'Shalve': [0.1, 0.1],
                'Vi': [1e6, 1e7],
                'h': [2, 4]
                },
        'Leakage': {
                'vl': [1e-9, 1e-9]
                },
        }
    return param

def libraries(nprom, nori):
    """ Define library values for:
        - Origin of replication
        - Promoters
    """
    
    param = {
        'Expression': 1e3*np.random.randint(1,1000,nprom),
        'Copy_number': np.random.randint(1,100,nori)
            }
    for y in param:
        param[y].sort()
    return param

def Parameters(nori,nprom,nsteps,nvariants):
    """ Define de parameters and ranges """
    """ TO DO: variants!! """
    par = {}
    plib = libraries(nprom,nori)
    par['Copy_number'] = plib['Copy_number']
    par['Expression'] = plib['Expression']
    par['Step'] = []
    for i in np.arange(nsteps):
        vals = []
        for j in np.arange(nvariants):
            vals.append( instance() )
        par['Step'].append( vals )
    return par
            
def Construct(par,design):
    promoters = []
    for x in np.arange(1,len(design),2):
        # Backbone promoter
        if x == 1:
            promoters.append( par['Expression'][design[x]] )
        # For the rest of promoters, we assume half of them empty
        else:
            if design[x]+1 > len(par['Expression']):
                promoters.append( None )
            else:
                promoters.append( par['Expression'][design[x]] )
    # Use the information about promoters to create the pathway  
    pw = pathway(promoters)
    initModel( pw, nsteps=len(par['Step']), substrate=1.0 )
    # Init model??
    # Set up the copy number
    for i in np.arange(len(par['Step'])):
        pw['m'+str(i+1)+'_Copy_number'] = float(par['Copy_number'][design[0]])
    # Set up the gene

    for i in np.arange(len(par['Step'])):
        enzyme = par['Step'][i][design[2+i*2]]
        for val in enzyme:
            (mean, std) = enzyme[val]
            p = np.random.normal( mean, std )
            param = 'm{}_{}'.format( i+1, val )
            pw[ param ] = p
    return pw
        

def instance():
    """ Generate an instance mean, std """
    par = ranges()
    vals = {}
    for group in par:
        for x in par[group]:
            xmax = par[group][x][1]
            xmin = par[group][x][0]
            mean = np.random.uniform( xmin, xmax )
            std = mean/100.0 + np.random.rand()*(xmax-xmin)/100.0
            vals['_'.join([group,x])] = ( mean,std )
    return vals
            
def initModel(model, substrate=0.0, nsteps=5):
    """ Each step in the pathway requires the following parameter definitions:
            - Induction: Shalve, Vi, h
            - Expression: k1
            - Degradation: k2
            - Leakage: vl
            - Catalysis: Km, V
            - Initial substrate concentration
            - Inducer concentrations
    """
    # Init all species to 0
    for step in np.arange(0,nsteps):
        model['m'+str(step+1)+'_Substrate'] = 0
        model['m'+str(step+1)+'_Enzyme'] = 0
        try:
            model['m'+str(step+1)+'_Inducer'] = 1
        except:
            pass
        try:
            model['m'+str(step+1)+'_Activated_promoter'] = 0
        except:
            pass
    model['m'+str(nsteps)+'_Product'] = 0
    model['m1_Substrate'] = substrate

    

class metPath:
    def __init__(self, steps):
        """ Init model and parameters """
        self.steps = steps
        self.model = modelTemplate( steps )
        self.vals = []
        for i in np.arange( steps ):
            self.vals.append( instance() )
    
    def sample(self, initSubstrate=1.0):
        """ Create a sample of the model 
            with given inital substrate concentration.
            Start inducers.
        """
        for i in np.arange( self.steps ):
            v = self.vals[i]
            for group in v:
                for x in v[group]:
                    (mean, std) = v[group][x]
                    p = np.random.normal( mean, std )
                    param = 'm{}_{}_{}'.format( i+1, group, x )
                    self.model[ param ] = p      
        initModel( self.model )
        self.model[ 'm1_Substrate' ] = initSubstrate
        for i in np.arange( self.steps ):
            induc = 'm{}_Inducer'.format(i+1)
            if induc in self.model:
                self.model[ induc ] = 1.0

def SelectCurves(pw):
    selections = []
    target = None
    for i in pw.timeCourseSelections:
        if i.endswith('Inducer]') or i.endswith('promoter]') or  i.endswith('Enzyme]') or i.endswith('Growth]'):
            continue
        selections.append(i)
        if i.endswith('Product]'):
            target = i
    pw.timeCourseSelections = selections
    pw.steadyStateSelections = selections
    return target
        
def SimulateDesign(steps=3, nplasmids=2, npromoters=2, variants=3, libsize=32, show=False):
    steps = steps
    variants = variants
    npromoters = npromoters
    nplasmids = nplasmids
    libsize = libsize
    positional = False
    par = Parameters(nplasmids,npromoters,steps,variants)
    diagnostics = evaldes( steps, variants, npromoters, nplasmids, libsize, positional )
    M = diagnostics['M']
    results = []
    for i in np.arange(M.shape[0]):
        design = M[i,:]
        pw = Construct(par,design)
        target = SelectCurves(pw)
        s = pw.simulate(0,400,1000)
        ds = pd.DataFrame(s,columns=s.colnames)
        results.append( s[target][-1] )
    if show:
        plt.figure(1)
        te.show()
    return pw, ds, M, results, par, diagnostics

# TO DO: multiple random sims per design? (but with same params)

def FitModel(M,results):
    columns = ['C'+str(i) for i in np.arange(M.shape[1])]
    dd = pd.DataFrame( M, columns=columns )
    promLevels = []
    for j in np.arange(3,dd.shape[1],2):
        promLevels.append( int(len(dd.iloc[:,j].unique())/2) )  
    for i in np.arange(M.shape[0]):
        for j in np.arange(M.shape[1]):
            # Add exception for promoters
            dd.iloc[i,j] = "L"+str(M[i,j])
            if j>2 and ( (j+1) % 2 == 0):
                plevel = promLevels[ int( (j-3)/2 ) ]
                if M[i,j] > plevel-1:
                    dd.iloc[i,j] = "L"+str(plevel)
                else:
                    dd.iloc[i,j] = "L"+str(M[i,j])
                
    dd['y'] = results
    formula = 'y ~ '+' + '.join(columns)
    ols = smf.ols( formula=formula, data=dd)
    res = ols.fit()
    return res, dd

def BestCombinations(res, dd):
    levels = []
    for j in np.arange(dd.shape[1]-1):
        levels.append( dd.iloc[:,j].unique() )
    comb = []
    for combo in product( *levels ):
        comb.append( combo )
    ndata = pd.DataFrame( comb, columns=dd.columns[0:-1] )
    ndata['pred'] = res.predict( ndata )
    ndata = ndata.sort_values(by='pred', ascending=False)
    ndata = ndata.reset_index(drop=True)
    return ndata

def ValidatePred(ndata, par, random=100):
    """ Simulating all combinations will become too expensive with large sets! """
    """ Alternative ask for a random sample """
    if random is None:
        points = np.arange(ndata.shape[0])
    else:
        points = np.hstack( [ [0,ndata.shape[0]-1], 
                             np.random.choice(ndata.shape[0],
                                              min(ndata.shape[0],random-2),
                                              replace=False) ] )
    results = []
    for i in points:
        design = [ int( re.sub('L', '',x)) for x in np.array( ndata.iloc[i,0:-1] )  ]

        pw = Construct(par,design)
        target = SelectCurves(pw)
        s = pw.simulate(0,400,1000)
        ds = pd.DataFrame(s,columns=s.colnames)
        results.append( s[target][-1] )
    ndata.loc[points,'sim'] = results
    ix = np.logical_not( np.isnan( ndata['sim'] ) )
    cc = np.corrcoef( ndata.loc[ix,'pred'], ndata.loc[ix,'sim'] )[0,1]
    rms = np.sqrt( np.sum((ndata.loc[ix,'pred'] - ndata.loc[ix,'sim'] )**2 )/len(np.where(ix)) )
    performance = { 'cc': cc, 'rms': rms }
    return performance
    
def PlotResults(ndata):
    plt.figure(2)
    plt.scatter( ndata['pred'],ndata['sim'] )
    plt.show()
    
def POC(steps=3, nplasmids=2, npromoters=2, variants=1, libsize=32, show=False):
    # Generate a DoE-based library and simulate results
    pw, ds, M, results, par, diagnostics = SimulateDesign(steps, nplasmids, npromoters, variants, libsize)
    # Fit a regression (constrast) model
    res, dd = FitModel(M,results)
    # Predict combinations based on the model
    ndata = BestCombinations(res,dd)
    # Validate predictions
    performance = ValidatePred(ndata, par)
    if show:
        PlotResults(ndata)
    return simInfo(diagnostics, performance)
    
def simInfo(diagnostics, performance, positional=False):
    head = ('steps', 'variants', 'npromoters', 'nplasmids', 'pos', 'libsize', 'eff', 'space', 'pow', 'rpv', 'cc', 'rms')
    steps = diagnostics['steps']
    variants = diagnostics['variants']
    npromoters = diagnostics['npromoters']
    nplasmids = diagnostics['nplasmids']
    libsize = diagnostics['libsize']
    J = diagnostics['J']
    pows = diagnostics['pow']
    rpvs = diagnostics['rpv']
    factors = diagnostics['factors']
    v = [len(x) for x in factors]
    if positional:
        pos = 1
    else:
        pos = 0
    try:
        pown = np.mean(pows)
    except:
        pown = 0
    try:
        rpvn = np.mean(rpvs)
    except:
        rpvn = np.nan
    cc = performance['cc']
    rms = performance['rms']
    row = (steps, variants, npromoters, nplasmids, pos, libsize, J, np.prod(v), pown, rpvn, cc, rms)
    return row


