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
          species Inducer in Cell, Activated_promoter in Cell;
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

          // Variable initializations:
          Induction_Shalve = 1e-1;
          Induction_Vi = 1e7;
          Induction_h = 1.85;
          Expression_k1 = 1e6;
          Leakage_vl = 1e-9;
          Degradation_k2 = 1e-6;
          Catalysis_Km = 0.1;
          Catalysis_kcat = 0.1;

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
                'Km': [0.001, 1e2],
                'kcat': [0.01, 10] 
                },
        'Degradation': {
                'k2': [10, 1e3]
                },
        'Expression': {
                'k1': [10, 1e3]
                },       
        'Induction': {
                'Shalve': [0.0001, 1],
                'Vi': [1e3, 1e5],
                'h': [2, 6]
                },
        'Leakage': {
                'vl': [1e-6, 1e-4]
                },
        }
    return param

def libraries(nprom, nori):
    """ Define library values for:
        - Origin of replication
        - Promoters
    """
    
    param = {
        'Expression': np.random.randint(1,1000,nprom),
        'Copy_number': np.random.randint(1,100,nori)
            }
    return param

def instance():
    """ Generate an instance mean, std """
    par = ranges()
    vals = {}
    for group in par:
        vals[group] = {}
        for x in par[group]:
            xmax = par[group][x][1]
            xmin = par[group][x][0]
            mean = np.random.uniform( xmin, xmax )
            std = mean/100.0 + np.random.rand()*(xmax-xmin)/100.0
            vals[group][x] = ( mean,std )
    return vals
            
def initModel(model, nsteps=5):
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
            model['m'+str(step+1)+'_Activated_promoter'] = 0
        except:
            pass
    model['m'+str(nsteps)+'_Product'] = 0
    

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
            self.model[ 'm{}_Inducer'.format(i+1) ] = 1.0


















