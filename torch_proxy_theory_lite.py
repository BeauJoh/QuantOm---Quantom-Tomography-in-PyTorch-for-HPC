import numpy as np
import torch

class TorchProxyTheoryLite(object):

    '''
    Theory class using a simplified, parametric PDF. This class was written, tested and reviewed by:
    - Daniel Lersch (dlersch@jlab.org)
    '''

    # Initialize:
    #*************************
    def __init__(self,config,devices="cpu"):
        self.devices = devices
        
        # True parameters unscaled (just so that we can look them up):
        self.p_true = config.get('p_true',[0.5,3.0,0.3,4.0,0.75])

        # Number of parameters and parameter scaling:
        self.nParameters = config.get('nParameters',5)
        self.parmin = config.get('parmin',[-0.5,2.75,0.0,3.0,0.0])
        self.parmax = config.get('parmax',[1.0,4.0,1.3,4.5,1.5],)
        
        # Transfer parameter scaling to torch tensors:
        self.parmin = torch.as_tensor(self.parmin,device=self.devices)
        self.parmax = torch.as_tensor(self.parmax,device=self.devices)
        
        # X-Section computation:
        self.n_cdf_points_x = config.get('n_cdf_points_x',10)
        self.n_cdf_points_y = config.get('n_cdf_points_y',10)
        self.n_points_x = config.get('n_points_x',10)
        self.n_points_y = config.get('n_points_y',10)
    
        self.x_min = config.get('x_min',0.001)
        self.x_max = config.get('x_max',0.999)
        self.y_min = config.get('y_min',0.001)
        self.y_max = config.get('y_max',0.999)

        # Take average of theory output:
        self.average = config.get('average',True)
    #*************************

    # Get the 'x-section':
    #*************************
    def get_xsection(self,p,x,y):
        # Rescale parameters first:
        rescaled_p = p * (self.parmax - self.parmin) + self.parmin

        x_dep = torch.pow(x,rescaled_p[0])*torch.pow((1.0-x),rescaled_p[1])
        y_dep = torch.pow(y,rescaled_p[2])*torch.pow((1.0-y),rescaled_p[3])

        if self.nParameters > 4:
           return x_dep * y_dep * (1.0 + rescaled_p[4]*(x*y))
        
        # We have the option to simplify the problem by leaving out the correlation term:
        return x_dep * y_dep * (x*y)
    #*************************
    
    # Generate a grid in xy:
    #*************************
    def gen_xy_grid(self):
        Ly = torch.linspace(self.y_min,self.y_max,self.n_points_y,device=self.devices)
        dLy = Ly[1:]-Ly[:-1]
        Lymid = 0.5*(Ly[1:]+Ly[:-1])
        Lymax = Ly[1:]
        Lymin = Ly[:-1]

        Lx = torch.linspace(self.x_min,self.x_max,self.n_points_x,device=self.devices)
        dLx = Lx[1:]-Lx[:-1]
        Lxmid = 0.5*(Lx[1:]+Lx[:-1])
        Lxmax = Lx[1:]
        Lxmin = Lx[:-1]

        Lxmid,Lymid=torch.meshgrid(Lxmid,Lymid,indexing='ij')
        Lxmin,Lymin=torch.meshgrid(Lxmin,Lymin,indexing='ij')
        Lxmax,Lymax=torch.meshgrid(Lxmax,Lymax,indexing='ij')
        dLx,dLy=torch.meshgrid(dLx,dLy,indexing='ij')

        return {
            'Lxmid': Lxmid,
            'Lxmin': Lxmin,
            'Lxmax': Lxmax,
            'Lymid': Lymid,
            'Lymin': Lymin,
            'Lymax': Lymax,
            'dLx': dLx,
            'dLy': dLy,
            'acc': torch.ones(dLx.size(),device=self.devices,dtype=torch.bool)
        }
    #*************************
    
    # Get the weights:
    #*************************
    def compute_weights(self,p):
        # Get the grid results first:
        results_from_grid = self.gen_xy_grid()
        
        # Determine an average x and Q2
        x_avg = results_from_grid['Lxmid']
        y_avg = results_from_grid['Lymid']

        # Get the differential cross section:
        diff_xsec = self.get_xsection(p,x_avg,y_avg)

        # Determine the integrand:
        integrand = diff_xsec*(x_avg * results_from_grid['dLx']) * (y_avg * results_from_grid['dLy']) * results_from_grid['acc']
        # And this leads to the total cross section:
        total_xsec = torch.sum(integrand)

        # Now compute the weights
        weights = integrand / total_xsec

        # Register everything into the dictionary that we already have:
        results_from_grid['total_xsec'] = total_xsec
        results_from_grid['weights'] = weights

        return results_from_grid
    #*************************

    # Compute x-sections on grid:
    #*************************
    def compute_xsec_on_grid(self,p):
        # Get the results from the grid and the weight calculation:
        results = self.compute_weights(p)
        grid = results['Lxmin'].shape

        # Cross section in x:
        u = torch.linspace(0,1,self.n_cdf_points_x,device=self.devices)
        x = ((results['Lxmin'][:,0].reshape(-1,1) + u * results['dLx'][:,0].reshape(-1,1))).flatten()

        xbins = x.reshape(-1,self.n_cdf_points_x)
        y = (results['Lymin'].T[:,0]).flatten()

        y,x=torch.meshgrid(y,x,indexing='ij')
        xsec_x = self.get_xsection(p,x,y).reshape(grid[1],-1,self.n_cdf_points_x)

        # Cross section in Q2:
        u = torch.linspace(0,1,self.n_cdf_points_y,device=self.devices)
        y = ((results['Lymin'].T[:,0].reshape(-1,1) + u * results['dLy'].T[:,0].reshape(-1,1))).flatten()
        ybins=y.reshape(-1,self.n_cdf_points_y)
        
        x = (results['Lxmin'][:,0]).flatten()
        x,y=torch.meshgrid(x,y,indexing='ij')
        xsec_y = self.get_xsection(p,x,y).reshape(grid[0],-1,self.n_cdf_points_y)

        w = results['weights']
        acc = results['acc']
        total_xsec = results['total_xsec']
        del results
        del u
        del x
        del y

        return xbins, xsec_x, ybins, xsec_y, w, acc, total_xsec
    #*************************

    # Now define the forward pass:
    #*************************
    def forward(self,params):
        xbins, xsec_x, ybins, xsec_y, weights, acceptance, total_xsec = torch.vmap(lambda p:self.compute_xsec_on_grid(p), in_dims=0, randomness="different")(params)
        
        if self.average == True:
            xbins_avg = torch.mean(xbins,0)
            xsec_x_avg = torch.mean(xsec_x,0)
            ybins_avg = torch.mean(ybins,0)
            xsec_y_avg = torch.mean(xsec_y,0)
            weights_avg = torch.mean(weights,0)
            acceptance_avg = torch.mean(acceptance.to(torch.float),0).to(torch.int)

            return torch.unsqueeze(xbins_avg,0), torch.unsqueeze(xsec_x_avg,0), torch.unsqueeze(ybins_avg,0), torch.unsqueeze(xsec_y_avg,0), torch.unsqueeze(weights_avg,0), torch.unsqueeze(acceptance_avg,0), total_xsec
        
        return xbins, xsec_x, ybins, xsec_y, weights, acceptance, total_xsec
    #*************************

    # Apply:
    #*************************  
    def apply(self,params):
        return self.forward(params)
    #*************************     
