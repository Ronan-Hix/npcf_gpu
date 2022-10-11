#########################################################################
#                    Ronan Hix 6 May 2022 
#          Based on code by Jiamin Hou Jan 26 2021         

#   The code is based on a set of isotropic function basis.
#   It is served as a small scale test of the NPCF functions.
#   N=2 ..5 are given in arXiv:2010.14418 A.1-A.4


#########################################################################

import sys, os, time, pickle
import collections, numpy
from scipy import spatial, stats
from itertools import combinations, tee, chain
import numpy as np
import math
import cupy as cp
from cupyx.profiler import benchmark, time_range
import tracemalloc

 
class calc_NPCF(object):
    
    def __init__(self, npcf=None, ngals=None, nbins=None, lbox=None, rmax=None, Nmax=None, 
                 numprimes=None, numseconds=None, lls=None, verbose=False, array_mode=True):
        
        self.npcf  = npcf
        self.ngals = ngals
        self.nbins = nbins
        self.lbox  = lbox
        self.rmax  = rmax
        self.verbose = verbose
        self.array_mode = array_mode
        self.Nmax = Nmax
        self.numprimes = numprimes
        self.numseconds = numseconds
        self.problems = 0
        
        if lls is not None:
            self.lls = lls
        else:
            if self.npcf == 3:
                self.lls = ['00', '11', '22']
            elif self.npcf == 4:
                self.lls = ['110', '111', '112', '222']
            elif self.npcf == 5:
                self.lls = ['11(0)11', '10(1)01', '21(1)12', '21(2)12',
                            '11(2)11', '11(1)11', '21(1)10', '21(3)12',
                            '21(2)11', '21(1)11']
            elif self.npcf == 6:
                self.lls = ['00(0)0(0)0', '11(0)0(0)0', '00(0)1(1)10','10(1)1(1)01', '10(1)1(2)02']
                       
    def init_coeff(self,Numblocks):
        
        '''initialize coefficients'''
        
        self.zeta = {}
        self.zeta_re = {}
        self.zeta_im = {}
        for il in self.lls:
            self.zeta[il] = cp.zeros([self.nbins]*int(self.npcf-1), dtype='c16')
            self.zeta_re[il] = cp.zeros([Numblocks]+[self.nbins]*int(self.npcf-1), dtype='f8')
            self.zeta_im[il] = cp.zeros([Numblocks]+[self.nbins]*int(self.npcf-1), dtype='f8')
            
    def save(self, sname, infile=None):
        
        '''save option:
        
           if no input: save coefficients as pickle file
        
           if input given: save as .txt file
        
        '''
        
        if infile is None:       
            with open(sname + 'mod.pkl', 'wb') as f:
                pickle.dump(self.zeta, f, -1)
            print(">> save file to", sname + 'mod.pkl')

                
        else:
            numpy.savetxt(sname + '.txt', infile)
            print(">> save file to", sname + '.txt')
        
            
    def make_catalog(self):
        
        '''
           make catalog based on the galcoords
           columns:
                   pos1, pos2, pos3, weight
        '''
        
        self.catalog = numpy.zeros([len(self.galcoords), 4])
        self.catalog[:,:3] = self.galcoords
        self.catalog[:,-1] = 1.


    def run(self):
        mempool = cp.get_default_memory_pool()
        print(f"Mem: {mempool.total_bytes()/1024/1024}")
        print(f"Mem: {mempool.used_bytes()/1024/1024}")
        numpy.random.seed(10)
        eps = 1e-8
        #print(cp.cuda.Device().attributes)
        
        # Define coords and computationally expensive combinatorics
        # that can be referenced by every primary galaxy loop
        self.galcoords = self.lbox * np.random.rand(self.ngals, 3)
        tree = spatial.cKDTree(self.galcoords,leafsize=self.lbox)
        galcoords_gpu = cp.array(self.galcoords, dtype=cp.float64)
        t1 = time.time()
        if self.Nmax < 30000:
            self.intdtype = 'int16'
        else:
            self.intdtype = 'int32'
            
        #Define custom vector data formats for use with CUDA Kernels
        self.int2 = np.dtype({'names': ['x','y'], 'formats': [np.int32]*2})
        self.int3 = np.dtype({'names': ['x','y','z'], 'formats': [np.int32]*3})
        self.short2 = np.dtype({'names': ['x','y'], 'formats': [np.int16]*2})
        self.short3 = np.dtype({'names': ['x','y','z'], 'formats': [np.int16]*3})
        self.double3 = np.dtype({'names': ['x','y','z'], 'formats': [np.float64]*3})
        if self.npcf == 3:
            self.shortvec = self.short2
            self.intvec = self.int2
            if self.numprimes == 1:
                self.ballctvec = self.short2
            else:
                self.ballctvec = self.int2
        if self.npcf == 4:
            self.shortvec = self.short3
            self.intvec = self.int3
            self.ballctvec = self.short3
            
            
        # Find all possible combinations for Nmax or total number
        # of neighbouring galaxies (whichever is smaller). 
        # masterballct is an object that contains a list of tuples of the indices of secondary galaxies
        tcomb1 = time.time()
        combo = chain.from_iterable(combinations(np.arange(self.nbins,dtype=self.intdtype), int(self.npcf-1)))
        sidecombos = np.fromiter(combo,dtype=self.intdtype).reshape(-1,int(self.npcf-1))
        tcomb2 = time.time()
        if self.npcf == 3:
                self.ballct_dtype='int16'
        if self.npcf == 4:
            self.ballct_dtype='int16'
        allsides = cp.asarray(sidecombos, dtype=self.ballct_dtype)
        print(f"tcomb: {tcomb2- tcomb1}")
        print(f"Memory (MB): {sidecombos.nbytes/1024/1024}")
        print(f"Mem Total: {mempool.total_bytes()/1024/1024}")
        print(f"Mem Used: {mempool.used_bytes()/1024/1024}")
        t2 = time.time()
        print(f"time: {t2-t1}")
        
        #Calculates the optimal number of blocks and assigns storage array
        self.Numblocks = math.ceil(sidecombos.shape[0]/64/(10**(self.npcf-3)))*self.numprimes
        print(f"Number of Blocks: {self.Numblocks}")
        self.init_coeff(self.Numblocks)
        
        
        
        #C++ Code for CUDA Kernels
        code_add='''
        __device__ double PI = 3.1415926535897932384626433832795028841971;
        
        extern "C" __global__ void add_func_3PCF(const short2* sides,
                                            int2 out_shape,
                                            int in_shape,
                                            int numprimes,
                                            int nbins,
                                            double3* binvals,
                                            double3* binvalssq,
                                            double3* binvalsmix,
                                            int* bincounts,
                                            double* out_00,
                                            double* out_11,
                                            double* out_22){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int k = i / in_shape;
            int j = i - k*in_shape;
            //printf(" (%d,%d,%d,%d,%d,%d,%d,%d)",i,k,j,in_shape*numprimes,sides[j].x + k*nbins,bincounts[sides[j].x + k*nbins],sides[j].y + k*nbins,bincounts[sides[j].y + k*nbins]);
            if (i < in_shape*numprimes
                && bincounts[sides[j].x + k*nbins] > 0
                && bincounts[sides[j].y + k*nbins] > 0
                ){
                    //printf(" [%d,%d,%d,%d,%d,%d,%d]",i,k,j,sides[j].x + k*nbins,bincounts[sides[j].x + k*nbins],sides[j].y + k*nbins,bincounts[sides[j].y + k*nbins]);
                    int loc = (sides[j].y+(out_shape.y*sides[j].x))+(out_shape.x*out_shape.y*k);
                    int inlocx = sides[j].x + k*nbins;
                    int inlocy = sides[j].y + k*nbins;
                    out_00[loc] += 
                        1./(4*PI)*bincounts[inlocx]*bincounts[inlocy];
                    out_11[loc] += 
                        -sqrtf(3.)/(4.*PI)*((binvals[inlocx].x*binvals[inlocy].x)+
                                            (binvals[inlocx].y*binvals[inlocy].y)+
                                            (binvals[inlocx].z*binvals[inlocy].z));
                    //binvalssq.x = R_xx, binvalsmix.x = R_xy, binvalsmix.y = R_xz, binvalsmix.z = R_yz,
                    out_22[loc] += 
                                    3./2.*sqrtf((5.)/(pow(4.*PI,2.)))*(
                                            (binvalssq[inlocx].x*binvalssq[inlocy].x)+
                                            (binvalssq[inlocx].y*binvalssq[inlocy].y)+
                                            (binvalssq[inlocx].z*binvalssq[inlocy].z)+
                                            2*(binvalsmix[inlocx].x*binvalsmix[inlocy].x)+
                                            2*(binvalsmix[inlocx].y*binvalsmix[inlocy].y)+
                                            2*(binvalsmix[inlocx].z*binvalsmix[inlocy].z)-
                                            1./3.*(bincounts[inlocx]*bincounts[inlocy]));
                }
            }       
        extern "C" __global__ void add_func_single_3PCF(const short2* sides,
                                            int2 out_shape,
                                            int in_shape,
                                            double3* binvals,
                                            double3* binvalssq,
                                            double3* binvalsmix,
                                            int* bincounts,
                                            double* out_00,
                                            double* out_11,
                                            double* out_22){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < in_shape
                && bincounts[sides[i].x] > 0
                && bincounts[sides[i].y] > 0
                ){
                    //printf(" [%d,%d,%d,%d]",sides[i].x,sides[i].y,bincounts[sides[i].x],bincounts[sides[i].y]);
                    
                    out_00[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)] += 
                        1./(4*PI)*bincounts[sides[i].x]*bincounts[sides[i].y];
                    out_11[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)] += 
                        -sqrtf(3.)/(4.*PI)*((binvals[sides[i].x].x*binvals[sides[i].y].x)+
                                            (binvals[sides[i].x].y*binvals[sides[i].y].y)+
                                            (binvals[sides[i].x].z*binvals[sides[i].y].z));
                    //binvalssq.x = R_xx, binvalsmix.x = R_xy, binvalsmix.y = R_xz, binvalsmix.z = R_yz,
                    out_22[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)] += 
                                    3./2.*sqrtf((5.)/(pow(4.*PI,2.)))*(
                                            (binvalssq[sides[i].x].x*binvalssq[sides[i].y].x)+
                                            (binvalssq[sides[i].x].y*binvalssq[sides[i].y].y)+
                                            (binvalssq[sides[i].x].z*binvalssq[sides[i].y].z)+
                                            2*(binvalsmix[sides[i].x].x*binvalsmix[sides[i].y].x)+
                                            2*(binvalsmix[sides[i].x].y*binvalsmix[sides[i].y].y)+
                                            2*(binvalsmix[sides[i].x].z*binvalsmix[sides[i].y].z)-
                                            1./3.*(bincounts[sides[i].x]*bincounts[sides[i].y]));
                }
            }

        '''
        code_manip='''
        __device__ double3 operator/(const double3& lhs, const double& divisor) {
            return make_double3(lhs.x / divisor,
                                lhs.y / divisor,
                                lhs.z / divisor);
        }
        __device__ double dot(const double3& lhs, const double3& rhs) {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }
         __device__ double norm(const double3& lhs){
            return sqrtf(dot(lhs,lhs));
        }  
        __device__ double3 normalize(const double3& lhs){
            return lhs/norm(lhs);
        }
        // Assigns galaxies to radial bins
        extern "C" __global__ void histogram(const double* rad_coord,
                                            double3* coords,
                                            int shape,
                                            int rmax,
                                            int nbins,
                                            int numseconds,
                                            double3* out1,
                                            double3* out2,
                                            double3* outmix,
                                            int* out_count){                                           
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if(i < shape
                && (fabsf(coords[i].x) > 0
                or fabsf(coords[i].y) > 0
                or fabsf(coords[i].z > 0))){

                coords[i] = normalize(coords[i]);
                int k = i / numseconds;
                int j = __double2int_rz(rad_coord[i]/rmax*nbins);
                atomicAdd(&out_count[j+k*nbins],1);
                
                atomicAdd(&out1[j+k*nbins].x,coords[i].x);
                atomicAdd(&out1[j+k*nbins].y,coords[i].y);
                atomicAdd(&out1[j+k*nbins].z,coords[i].z);
                
                atomicAdd(&out2[j+k*nbins].x,coords[i].x*coords[i].x);
                atomicAdd(&out2[j+k*nbins].y,coords[i].y*coords[i].y);
                atomicAdd(&out2[j+k*nbins].z,coords[i].z*coords[i].z);
                
                atomicAdd(&outmix[j+k*nbins].x,coords[i].x*coords[i].y);
                atomicAdd(&outmix[j+k*nbins].y,coords[i].x*coords[i].z);
                atomicAdd(&outmix[j+k*nbins].z,coords[i].y*coords[i].z);
            }
        }

        '''
        
        P_lambda_code = '''
        '''
        #Define Kernel Functions for later use
        manip_module = cp.RawModule(code=code_manip)       
        add_module = cp.RawModule(code=code_add)
        P_lambda_module = cp.RawModule(code=P_lambda_code)
        if self.npcf == 3:
            if self.numprimes == 1:
                self.hist = manip_module.get_function('histogram')
                self.N3_and_add = add_module.get_function('add_func_3PCF')
            else:
                self.hist = manip_module.get_function('histogram')
                self.N3_and_add = add_module.get_function('add_func_3PCF')
        elif self.npcf == 4:
            self.ballct_func = manip_module.get_function('get_ballct_4PCF')
            self.sides_func = manip_module.get_function('get_sides_4PCF')
            self.N4_and_add = P_lambda_module.get_function('N4_coeff_add')
            
        else:
            print('Stop That')
        
        self.sides = []
        print(f"Nums: {type(self.numprimes//self.ngals)}")
        masterprime = np.arange(self.numprimes)*(self.ngals//self.numprimes)
        coord_lengths_to_pass = cp.zeros(self.numprimes).astype('int16')
        '''
        running_lengths = cp.zeros(self.ngals).astype('int32')       
        second_coord_3d_master = cp.array([[],[],[]]).reshape((0,3))
        second_coord_radial_master = cp.array([])
        #primes=masterprime+ii
        for n in range(self.ngals):
                print(f"{n}")
                second_coord_3d,second_coord_radial = self.get_coords(n,eps,tree,galcoords_gpu)
                second_coord_3d_master = cp.concatenate((second_coord_3d_master,second_coord_3d))
                second_coord_radial_master = cp.concatenate((second_coord_radial_master,second_coord_radial))
                coord_lengths[n] = len(second_coord_radial)
                running_lengths[n] = running_lengths[n-1]+len(second_coord_radial)
        '''
        #print(masterprime)
        for ii in range(self.ngals//self.numprimes): 
            if self.numprimes == 1:
                print(f">> sit on {ii}th galaxy")
                second_coord_3d_to_pass,second_coord_radial_to_pass = self.get_coords(ii,eps,tree,galcoords_gpu) 
                coord_lengths_to_pass[0] = len(second_coord_radial_to_pass)
                '''
                print(running_lengths[ii],ii)
                second_coord_3d_to_pass = second_coord_3d_master[running_lengths[ii]-coord_lengths[ii]:running_lengths[ii]]
                second_coord_radial_to_pass = second_coord_radial_master[running_lengths[ii]-coord_lengths[ii]:running_lengths[ii]]
                coord_lengths_to_pass = coord_lengths[ii]
                '''
            else:
                #second_coord_3d_to_pass = cp.array([[],[],[]]).reshape((0,3))
                second_coord_3d_to_pass = cp.zeros(3*self.numseconds*self.numprimes).reshape((self.numseconds*self.numprimes,3)) #Creates scaffold array for coordinates
                second_coord_radial_to_pass = cp.zeros(self.numseconds*self.numprimes)
                primes=masterprime+ii
                print(f">> sit on {primes}th galaxy")
                #second_coord_3d,second_coord_radial = self.get_coords(primes[0],eps,tree,galcoords_gpu)
                for n, prime in enumerate(primes):
                    second_coord_3d,second_coord_radial = self.get_coords(prime,eps,tree,galcoords_gpu)
                    num = len(second_coord_radial)
                    if num > self.numseconds:
                        print("Error: More secondaries found for galaxy {prime} than the secondary cap. Please increase secondary cap.")
                    coord_lengths_to_pass[n] = num
                    second_coord_3d_to_pass[n*self.numseconds:n*self.numseconds+num] = second_coord_3d
                    second_coord_radial_to_pass[n*self.numseconds:n*self.numseconds+num] = second_coord_radial
                    

            self.run_some_gals(ii,second_coord_3d_to_pass,second_coord_radial_to_pass,allsides,coord_lengths_to_pass)
            print(benchmark(self.get_coords,(ii,eps,tree,galcoords_gpu), n_repeat=10,n_warmup=10))
            print(benchmark(self.run_some_gals,(ii,second_coord_3d_to_pass,second_coord_radial_to_pass,allsides,coord_lengths_to_pass), n_repeat=10,n_warmup=10))
        for il in self.lls:
            self.zeta[il] = cp.sum(self.zeta_re[il],axis=0) + 1j*cp.sum(self.zeta_im[il],axis=0)
        print(f"Problems Encountered: {self.problems}")
        print(self.zeta)

        
    def get_coords(self,ii,eps,tree,galcoords_gpu):    
        prime_coord_3d = galcoords_gpu[ii]
        # ballct: list of indices of the secondary galaxies of the current primary galaxy
        ballct = tree.query_ball_point(self.galcoords[ii], self.rmax+eps)
        ballct.remove(ii)
        # coordinates of the secondary galaxies 
        second_coord_3d = galcoords_gpu[ballct] - prime_coord_3d   
        second_coord_radial = np.sqrt(np.sum(second_coord_3d**2, axis=1))
        
        # sort the secondary galaxies according to their distance
        '''
        sort = second_coord_radial.argsort()
        second_coord_3d = second_coord_3d[sort]
        second_coord_radial = second_coord_radial[sort]
        '''
        #print(type(second_coord_3d))
        return (second_coord_3d,second_coord_radial)
    
    def run_some_gals(self,ii,second_coord_3d,second_coord_radial,allsides,coord_lengths):
        #print(f">> also sit on {ii+self.ngals/2}th galaxy")
        #mempool = cp.get_default_memory_pool()
        #print(f"Mem tot: {mempool.total_bytes()/1024/1024}")
        #print(f"Mem used: {mempool.used_bytes()/1024/1024}")

        #ballct,second_coord_3d,second_coord_radial = self.get_coords(ii,eps,tree,galcoords_gpu)
        
        if second_coord_3d.shape[0] >= (self.npcf-1): #Excludes primaries with too few secondaries
            
            print(f"Num Secondaries: {second_coord_3d.shape[0]}")



            

        
            #Gets the radial bin index of the N-1 secondary galaxies
            #histx, bins = cp.histogram(second_coord_radial,bins=self.nbins,range = (0,self.rmax),weights=second_coord_3d[:,0])
            #histy, _ = cp.histogram(second_coord_radial,bins=self.nbins,range = (0,self.rmax),weights=second_coord_3d[:,1])
            #histz, _ = cp.histogram(second_coord_radial,bins=self.nbins,range = (0,self.rmax),weights=second_coord_3d[:,2])
            
            
            counts = cp.zeros(self.numprimes*self.nbins, dtype='int32')  
            hist1 = cp.zeros(self.numprimes*self.nbins*3, dtype='float64').reshape(self.numprimes,self.nbins,3).view(self.double3)
            hist2 = cp.zeros(self.numprimes*self.nbins*3, dtype='float64').reshape(self.numprimes,self.nbins,3).view(self.double3)
            histmix = cp.zeros(self.numprimes*self.nbins*3, dtype='float64').reshape(self.numprimes,self.nbins,3).view(self.double3)
            second_coord_3d = second_coord_3d.view(self.double3)
            nhistblocks = math.ceil(len(second_coord_radial)/1024)
            #print(nhistblocks)
            self.hist((nhistblocks,), (1024,),(second_coord_radial,second_coord_3d,
                                            len(second_coord_radial),self.rmax,self.nbins,self.numseconds,
                                            hist1,hist2,histmix,counts))
            #unique, counts = cp.unique(sides,return_counts=True)
            #print(second_coord_radial)
            #print(second_coord_3d[:,0])
            #print(counts)

            #print(histx)
            #print(allsides.shape)
            #print(f"Storage array: {self.zeta_re['110'].nbytes/1024/1024}")
            #print(f"Sides: {sides.nbytes/1024/1024}")
            #print(f"Mem tot: {mempool.total_bytes()/1024/1024}")
            #print(f"Mem used: {mempool.used_bytes()/1024/1024}")

            second_coord_3d = second_coord_3d.view(self.double3)   
            if self.npcf ==3:
                #Shape of the output array - used for indexing in function call
                allsides = allsides.view(self.short2)
                shape = np.asarray([self.zeta_re[self.lls[0]].shape[1],self.zeta_re[self.lls[0]].shape[2]]).astype(np.int32).view(self.int2)
                #Calculates coefficients and adds them to zeta
                #print(self.Numblocks)
                self.N3_and_add((self.Numblocks,), (64,),(allsides,shape,allsides.shape[0],self.numprimes,self.nbins,hist1,hist2,histmix,counts,
                                                    self.zeta_re['00'],self.zeta_re['11'],self.zeta_re['22'],
                                                    ))
            elif self.npcf == 4:
                print("No")

                #print(f"Mem tot: {mempool.total_bytes()/1024/1024}")
                #print(f"Mem used: {mempool.used_bytes()/1024/1024}")
        #                if self.verbose: print("    order", il, "coefficients",new_zeta)
        #         self.zeta = numpy.average(numpy.array(list(self.zeta.values())),axis=0).real
            #print(self.zeta_re['22'])


if __name__ == "__main__":

    start_time = time.time()

    npcf = 3
    ngals = 1000000
    nbins = 10
    lbox = 20
    rmax = 2
    Nmax = 5000
    numprimes = 40
    numseconds = np.minimum(Nmax,ngals) #ngals
    #Numblocks = 306000
    # lls_5pcf = ['11(0)11', '21(1)12']
    verbose=False
    
    zetas = calc_NPCF(npcf, ngals, nbins, lbox, rmax, Nmax, 
                      numprimes, numseconds, lls=None, verbose=verbose)
    #zetas.init_coeff(Numblocks)
    zetas.run()
    sdir = "/home/ronanhix/orange/"
    sname1 = sdir+f"results/coeff_{npcf:0d}pcf_ngals{ngals:0d}_nbins{nbins:0d}_lbox{lbox:0d}_rmax{rmax:0d}"
    zetas.save(sname1)
    sname2 = sdir+f"data/catalog_ngals{ngals:0d}_nbins{nbins:0d}_lbox{lbox:0d}_rmax{rmax:0d}"
    zetas.make_catalog()
    zetas.save(sname2, zetas.catalog)  
    #print(benchmark(zetas.run, n_repeat=2))
    elapsed_time = time.time() - start_time
    print(f"number of galaxies: {ngals:1d}")
    print(f"elapsed time: {elapsed_time:.3f} s")
