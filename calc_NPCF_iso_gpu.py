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

def P_lambda_gpu(coords, order='11(0)11', npcf=3):
    '''
        define basis function:
        
        Slepian&Cahn 2020 Appendix A
    '''
    if npcf == 3:
        
        r1_hat = (coords[:,0,:].T/cp.linalg.norm(coords[:,0,:], ord=2, axis=1)).T
        r2_hat = (coords[:,1,:].T/cp.linalg.norm(coords[:,1,:], ord=2, axis=1)).T
        
        if order == '00':
            return 1./4/np.pi * cp.ones(r1_hat.shape[0])
        
        elif order == '11':
            return -cp.sqrt(3)/4./np.pi * cp.sum(r1_hat*r2_hat, axis=1)
            
        elif order == '22':
            return 3./2*cp.sqrt(5./(4*np.pi)**2) * (cp.sum(r1_hat*r2_hat, axis=1)**2 - 1./3)
           
    elif npcf == 4:
        r1_hat = (coords[:,0,:].T/cp.linalg.norm(coords[:,0,:], ord=2, axis=1)).T
        r2_hat = (coords[:,1,:].T/cp.linalg.norm(coords[:,1,:], ord=2, axis=1)).T
        r3_hat = (coords[:,2,:].T/cp.linalg.norm(coords[:,2,:], ord=2, axis=1)).T

        if order == '110':
            return -1./4/np.pi*cp.sqrt(3./4/np.pi) * cp.sum(r1_hat*r2_hat, axis=1)

        elif order == '111':
            return -3j/cp.sqrt(2)/(4 * cp.pi)**(3./2) * cp.sum(r1_hat * cp.cross(r2_hat, r3_hat), axis=1)

        elif order == '112':
            return cp.sqrt(27/2/(4*cp.pi)**3) * (cp.sum(r1_hat * r3_hat, axis=1)*cp.sum(r2_hat * r3_hat, axis=1)-
                                                      1./3 * cp.sum(r1_hat * r2_hat, axis=1))
        elif order == '222':
            return -45/cp.sqrt(14*(4*cp.pi)**3) * (cp.sum(r1_hat * r2_hat, axis=1)*cp.sum(r1_hat * r3_hat, axis=1)*cp.sum(r2_hat * r3_hat, axis=1)
                                                          -1./3*cp.sum(r1_hat * r2_hat, axis=1)**2
                                                          -1./3*cp.sum(r1_hat * r3_hat, axis=1)**2
                                                          -1./3*cp.sum(r2_hat * r3_hat, axis=1)**2 + 2./9)
    elif npcf == 5:
        r1_hat = (coords[:,0,:].T/cp.linalg.norm(coords[:,0,:], ord=2, axis=1)).T
        r2_hat = (coords[:,1,:].T/cp.linalg.norm(coords[:,1,:], ord=2, axis=1)).T
        r3_hat = (coords[:,2,:].T/cp.linalg.norm(coords[:,2,:], ord=2, axis=1)).T
        r4_hat = (coords[:,3,:].T/cp.linalg.norm(coords[:,3,:], ord=2, axis=1)).T
        
        if order == '11(0)11':
            return 3./(4*np.pi)**2 * cp.sum(r1_hat * r2_hat, axis = 1) * cp.sum(r3_hat * r4_hat, axis = 1)

        elif order == '10(1)01':
            return -cp.sqrt(3)/(4*np.pi)**2 * cp.sum(r1_hat * r4_hat, axis = 1)
        elif order == '21(1)12':
            return -9*cp.sqrt(3)/(2*(4*np.pi)**2) * (cp.sum(r1_hat * r2_hat, axis = 1)*cp.sum(r1_hat * r4_hat, axis = 1)*cp.sum(r3_hat * r4_hat, axis = 1)
                                                        -1./3*cp.sum(r2_hat * r4_hat, axis = 1)*cp.sum(r3_hat * r4_hat, axis = 1)
                                                        -1./3*cp.sum(r1_hat * r2_hat, axis = 1)*cp.sum(r1_hat * r3_hat, axis = 1)
                                                        +1./9*cp.sum(r2_hat * r3_hat, axis = 1))
        elif order == '21(2)12':
            return 3.*cp.sqrt(5)/(2*(4*np.pi)**2)*((2*cp.sum(r1_hat * r4_hat, axis = 1))*(cp.sum(r1_hat * r4_hat, axis = 1)*cp.sum(r2_hat * r3_hat, axis = 1)
                                                    -cp.sum(r1_hat * r3_hat, axis = 1)*cp.sum(r2_hat * r4_hat, axis = 1))
                                                    -cp.sum(r1_hat * r4_hat, axis = 1)*cp.sum(r1_hat * r2_hat, axis = 1)*cp.sum(r3_hat * r4_hat, axis = 1)
                                                    +cp.sum(r2_hat * r4_hat, axis = 1)*cp.sum(r3_hat * r4_hat, axis = 1)
                                                    +cp.sum(r1_hat * r2_hat, axis = 1)*cp.sum(r1_hat * r3_hat, axis = 1)
                                                    -cp.sum(r2_hat * r3_hat, axis = 1))
        elif order == '11(2)11':
            return 9./(2*cp.sqrt(5)*(4*np.pi)**2)*(cp.sum(r1_hat * r3_hat, axis = 1)*cp.sum(r2_hat * r4_hat, axis = 1)
                                                    +cp.sum(r1_hat * r4_hat, axis = 1)*cp.sum(r2_hat * r3_hat, axis = 1)
                                                    -2./4*cp.sum(r1_hat * r2_hat, axis = 1)*cp.sum(r3_hat * r4_hat, axis = 1))
        elif order == '11(1)11':
            return 3*cp.sqrt(3)/(2*(4*np.pi)**2)*(cp.sum(r1_hat * r3_hat, axis = 1)*cp.sum(r2_hat * r4_hat, axis = 1)
                                                    -cp.sum(r1_hat * r4_hat, axis = 1)*cp.sum(r2_hat * r3_hat, axis = 1))
        elif order == '21(1)10':
            return 3*cp.sqrt(3)/(np.sqrt(2)*(4*np.pi)**2)*(cp.sum(r1_hat * r2_hat, axis = 1)*cp.sum(r1_hat * r3_hat, axis = 1)
                                                            -1./3*cp.sum(r2_hat * r3_hat, axis = 1))
        elif order == '21(3)12':
            return -15./(2*np.sqrt(7)*(4*np.pi)**2) * (cp.sum(r1_hat * r4_hat, axis = 1)**2 * cp.sum(r2_hat * r3_hat, axis = 1)
                                                    -4/5*cp.sum(r1_hat * r2_hat, axis = 1)*cp.sum(r1_hat * r4_hat, axis = 1)*cp.sum(r3_hat * r4_hat, axis = 1)
                                                    +2*cp.sum(r1_hat * r3_hat, axis = 1)*cp.sum(r1_hat * r4_hat, axis = 1)*cp.sum(r2_hat * r4_hat, axis = 1)
                                                    -2/5*cp.sum(r1_hat * r2_hat, axis = 1)*cp.sum(r1_hat * r3_hat, axis = 1)
                                                    -2/5*cp.sum(r2_hat * r4_hat, axis = 1)*cp.sum(r3_hat * r4_hat, axis = 1)
                                                    -1/5*cp.sum(r2_hat * r3_hat, axis = 1))
        elif order == '21(2)11':
            return 3j*cp.sqrt(3)/(2*(4*np.pi)**2)*(cp.sum((cp.sum(r1_hat * r3_hat, axis = 1)*r4_hat.T).T * cp.cross(r1_hat,r2_hat), axis=1)
                                                    +cp.sum((cp.sum(r1_hat * r4_hat, axis = 1)*r3_hat.T).T * cp.cross(r1_hat,r2_hat), axis=1))
        elif order == '21(1)11':
            return 9j/(2*(4*cp.pi)**2)*(cp.sum((cp.sum(r1_hat * r2_hat, axis = 1)*r1_hat.T).T * cp.cross(r3_hat,r4_hat), axis=1)
                                        -1/3*cp.sum(r2_hat * cp.cross(r3_hat,r4_hat), axis=1))
    '''
    elif npcf == 6:
        r1_hat = rhats[0]
        r2_hat = rhats[1]
        r3_hat = rhats[2]
        r4_hat = rhats[3]
        r5_hat = rhats[4]
        r6_hat = rhats[5]
        
        if order == '00(0)0(0)0':
            return 1./(2*numpy.sqrt(numpy.pi))**5

        elif order == '11(0)0(0)0':
            return -numpy.sqrt(3)/32/(numpy.pi)**(5./2) * numpy.dot(r1_hat, r2_hat)

        elif order == '00(0)1(1)10':
            return -numpy.sqrt(3)/32/(numpy.pi)**(5./2) * numpy.dot(r3_hat, r4_hat)

        elif order == '10(1)1(1)01':
            return -1.5*numpy.sqrt(2)/32/(numpy.pi)**(5./2) * numpy.dot(r1_hat, (numpy.cross(r5_hat, r3_hat)))
        
        elif order == '10(1)1(2)02':
            return (numpy.sqrt(6)/64/(numpy.pi)**(5./2)
                    *(2*numpy.dot(r1_hat, r3_hat) * numpy.dot(r5_hat, r5_hat)
                    +3*numpy.dot(r3_hat, numpy.cross(r5_hat, numpy.cross(r5_hat, r1_hat))) ))
        '''




class calc_NPCF(object):
    
    def __init__(self, npcf=None, ngals=None, nbins=None, lbox=None, rmax=None, Nmax=None, numprimes=None, lls=None, verbose=False, array_mode=True):
        
        self.npcf  = npcf
        self.ngals = ngals
        self.nbins = nbins
        self.lbox  = lbox
        self.rmax  = rmax
        self.verbose = verbose
        self.array_mode = array_mode
        self.Nmax = Nmax
        self.numprimes = numprimes
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
        combo = chain.from_iterable(combinations(np.arange(np.min([self.Nmax,len(self.galcoords)]),dtype=self.intdtype), int(self.npcf-1)))
        masterballct_cpu = np.fromiter(combo,dtype=self.intdtype).reshape(-1,int(self.npcf-1))
        tcomb2 = time.time()
        if self.npcf == 3:
            if self.numprimes == 1:
                self.ballct_dtype='int16'
            else:
                self.ballct_dtype='int32'
        if self.npcf == 4:
            self.ballct_dtype='int16'
        masterballct = cp.asarray(masterballct_cpu, dtype=self.ballct_dtype)
        print(f"tcomb: {tcomb2- tcomb1}")
        print(f"Memory (MB): {masterballct_cpu.nbytes/1024/1024}")
        print(f"Mem Total: {mempool.total_bytes()/1024/1024}")
        print(f"Mem Used: {mempool.used_bytes()/1024/1024}")
        t2 = time.time()
        print(f"time: {t2-t1}")
        
        #Calculates the optimal number of blocks and assigns storage array
        self.Numblocks = math.ceil(masterballct.shape[0]/1024/(10**(self.npcf-3)))*self.numprimes+1
        print(f"Number of Blocks: {self.Numblocks}")
        self.init_coeff(self.Numblocks)
        
        
        
        #C++ Code for CUDA Kernels
        code_add='''
        #include <cupy/complex.cuh>
        extern "C" __global__ void add_func_3PCF(const short2* sides,
                                            int2 out_shape,
                                            int in_shape,
                                            complex<double>* new_zeta,
                                            double* out_re,
                                            double* out_im){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < in_shape 
                && 0 < fabsf(sides[i].y+(out_shape.y*sides[i].x)) 
                && fabsf(sides[i].y+(out_shape.y*sides[i].x)) < (out_shape.x*out_shape.y)
                ){
                
                if (sides[i].x != sides[i].y){
                    atomicAdd(&out_re[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)],new_zeta[i].real());
                    atomicAdd(&out_im[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)],new_zeta[i].imag());
                }
            }       
        }
        extern "C" __global__ void add_func_4PCF(const short3* sides,
                                            int3 out_shape,
                                            int in_shape,
                                            complex<double>* new_zeta,
                                            double* out_re,
                                            double* out_im){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i <= in_shape 
                && 0 < fabsf((sides[i].z+(out_shape.z*sides[i].y)+(out_shape.z*out_shape.y*sides[i].x))) 
                && fabsf((sides[i].z+(out_shape.z*sides[i].y)+(out_shape.z*out_shape.y*sides[i].x))) < (out_shape.x*out_shape.y*out_shape.z)
                ){

                if (sides[i].x != sides[i].y
                  && sides[i].x != sides[i].z
                  && sides[i].y != sides[i].z){

                    atomicAdd(&out_re[(sides[i].z+(out_shape.z*sides[i].y)+(out_shape.z*out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*out_shape.z*blockIdx.x)],new_zeta[i].real());
                    atomicAdd(&out_im[(sides[i].z+(out_shape.z*sides[i].y)+(out_shape.z*out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*out_shape.z*blockIdx.x)],new_zeta[i].imag());
                    }

            }       
        }
        '''
        code_manip='''
        __device__ short2 operator+(const short2& lhs, const short& rhs) {
            return make_short2(lhs.x + rhs,
                                lhs.y + rhs);
        }
        __device__ short2 operator+(const short2& lhs, const int& rhs) {
            return make_short2(lhs.x + rhs,
                                lhs.y + rhs);
        }
        __device__ int2 operator+(const int2& lhs, const short& rhs) {
            return make_int2(lhs.x + rhs,
                                lhs.y + rhs);
        }
        __device__ int2 operator+(const int2& lhs, const int& rhs) {
            return make_int2(lhs.x + rhs,
                                lhs.y + rhs);
        }
        // Returns only partner pairs with partners that are
        // in the sphere containing neighbors
        extern "C" __global__ void get_ballct_3PCF(const int2* masterballct,
                                            int in_shape,
                                            int num_primes,
                                            short* prime_sizes,
                                            int n_neighbors,
                                            int2* out){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (i < in_shape*num_primes){
                int j = i / in_shape;
                int offset = 0;
                int k=0;
                for(k=0;k<j;k++){
                    offset += prime_sizes[k];
                }
                
                if (i < in_shape*(j+1)
                    && masterballct[i-in_shape*j].x < prime_sizes[j] 
                    && masterballct[i-in_shape*j].y < prime_sizes[j]){
                    
                    
                    out[i] = masterballct[i-in_shape*j] + offset;
                    }
                }
                
                //if (i < in_shape
                   // && masterballct[i].x < prime_sizes[0] 
                   // && masterballct[i].y < prime_sizes[0]){
                    //printf(" [%d,%d] ",i,in_shape);
                   // out[i] = masterballct[i] ;
              //  }
               // else if (i >= in_shape
                //    && masterballct[i-in_shape].x < prime_sizes[1] 
               //     && masterballct[i-in_shape].y < prime_sizes[1]){
                    //printf(" [%d,%d,%d,%d] ",i,i-in_shape,masterballct[i-in_shape].x,masterballct[i-in_shape].y);
                //    out[i] = masterballct[i-in_shape] + prime_sizes[0];
               // }
                        
        }
        // Returns only partner pairs with partners that are
        // in the sphere containing neighbors - For single primary
        extern "C" __global__ void get_ballct_3PCF_single(const short2* masterballct,
                                            int in_shape,
                                            int num_primes,
                                            short* prime_sizes,
                                            int n_neighbors,
                                            short2* out){
            int i = blockDim.x * blockIdx.x + threadIdx.x;           
            if (i < in_shape){     
                if (i < in_shape
                    && masterballct[i].x < prime_sizes[0] 
                    && masterballct[i].y < prime_sizes[0]){                 
                        out[i] = masterballct[i];
                    }
                }
                        
        }
        
        // Returns only partner triplets with partners that are
        // in the sphere containing neighbors
        extern "C" __global__ void get_ballct_4PCF(const short3* masterballct,
                                            int in_shape,
                                            int n_neighbors,
                                            short3* out){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < in_shape 
                && masterballct[i].x < n_neighbors 
                && masterballct[i].y < n_neighbors
                && masterballct[i].z < n_neighbors){
                
                out[i] = masterballct[i] ;
            }            
        }
        // Assigns side pairs to radial bins
        extern "C" __global__ void get_sides_3PCF(const double* rad_coord,
                                            int2* index,
                                            int shape,
                                            int rmax,
                                            int nbins,
                                            short2* out){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if(i < shape){
                out[i].x = __double2float_rz(rad_coord[index[i].x]/rmax*nbins);
                out[i].y = __double2float_rz(rad_coord[index[i].y]/rmax*nbins);;
            }
        }
        // Assigns side pairs to radial bins
        extern "C" __global__ void get_sides_3PCF_single(const double* rad_coord,
                                            short2* index,
                                            int shape,
                                            int rmax,
                                            int nbins,
                                            short2* out){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if(i < shape){
                out[i].x = __double2float_rz(rad_coord[index[i].x]/rmax*nbins);
                out[i].y = __double2float_rz(rad_coord[index[i].y]/rmax*nbins);;
            }
        }
        
        // Assigns side triplets to radial bins
        extern "C" __global__ void get_sides_4PCF(const double* rad_coord,
                                            short3* index,
                                            int shape,
                                            int rmax,
                                            int nbins,
                                            short3* out){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if(i <= shape){
                //printf(" [%d,%d] ", );
                out[i].x = __double2float_rz(rad_coord[index[i].x]/rmax*nbins);
                out[i].y = __double2float_rz(rad_coord[index[i].y]/rmax*nbins);
                out[i].z = __double2float_rz(rad_coord[index[i].z]/rmax*nbins);
            }
        }
        '''
        
        P_lambda_code = '''
        #include <cupy/complex.cuh>
        //Basic Functions
        __device__ double PI = 3.1415926535897932384626433832795028841971;
        __device__ double3 operator+(const double3& lhs, const double3& rhs) {
            return make_double3(lhs.x + rhs.x,
                                lhs.y + rhs.y,
                                lhs.z + rhs.z);
        }
        __device__ double3 operator/(const double3& lhs, const double& divisor) {
            return make_double3(lhs.x / divisor,
                                lhs.y / divisor,
                                lhs.z / divisor);
        }
        __device__ double dot(const double3& lhs, const double3& rhs) {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }
        __device__ double3 cross(const double3& lhs, const double3& rhs) {
            return make_double3(lhs.y*rhs.z - lhs.z*rhs.y, 
                                lhs.z*rhs.x - lhs.x*rhs.z, 
                                lhs.x*rhs.y - lhs.y*rhs.x);
        }
        __device__ double norm(const double3& lhs){
            return sqrtf(dot(lhs,lhs));
        }  
        __device__ double3 normalize(const double3& lhs){
            return lhs/norm(lhs);
        }
        
        //3PCF Definitions 
        //Calculates 3PCF coefficents and adds them to appropriate output array
        extern "C" __global__ void N3_coeff_add(const double3* coords,
                                                    int2* indices,
                                                    short2* sides,
                                                    double* out_00_re,
                                                    double* out_11_re,
                                                    double* out_22_re,            
                                                    int in_shape,
                                                    int2 out_shape) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < in_shape
             && sides[i].x != sides[i].y){
             
                double3 r1_hat = normalize(coords[indices[i].x]);
                double3 r2_hat = normalize(coords[indices[i].y]);
                
                atomicAdd(&out_00_re[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)],1./(4*PI));
                atomicAdd(&out_11_re[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)],-sqrtf(3.)/(4.*PI)*dot(r1_hat,r2_hat));
                atomicAdd(&out_22_re[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)],(3./2.)*sqrtf(5./pow(4.*PI,2.))*(pow(dot(r1_hat,r2_hat),2.) - 1./3.));
            }
        }
        //Calculates 3PCF coefficents and adds them to appropriate output array - For use with single primary at a time
        extern "C" __global__ void N3_coeff_add_single(const double3* coords,
                                                    short2* indices,
                                                    short2* sides,
                                                    double* out_00_re,
                                                    double* out_11_re,
                                                    double* out_22_re,            
                                                    int in_shape,
                                                    int2 out_shape) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < in_shape
             && sides[i].x != sides[i].y){
             
                double3 r1_hat = normalize(coords[indices[i].x]);
                double3 r2_hat = normalize(coords[indices[i].y]);
                
                atomicAdd(&out_00_re[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)],1./(4*PI));
                atomicAdd(&out_11_re[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)],-sqrtf(3.)/(4.*PI)*dot(r1_hat,r2_hat));
                atomicAdd(&out_22_re[(sides[i].y+(out_shape.y*sides[i].x))+(out_shape.x*out_shape.y*blockIdx.x)],(3./2.)*sqrtf(5./pow(4.*PI,2.))*(pow(dot(r1_hat,r2_hat),2.) - 1./3.));
            }
        }
        //4PCF Definitions
        //Calculates 4PCF coefficents and adds them to appropriate output array
        extern "C" __global__ void N4_coeff_add(const double3* coords,
                                                    short3* indices,
                                                    short3* sides,
                                                    double* out_110_re,
                                                    double* out_111_im,
                                                    double* out_112_re,
                                                    double* out_222_re,             
                                                    int in_shape,
                                                    int3 out_shape) {
                                                    
          int i = blockDim.x * blockIdx.x + threadIdx.x;
          if (i < in_shape
              && sides[i].x != sides[i].y
                  && sides[i].x != sides[i].z
                  && sides[i].y != sides[i].z){

              double3 r1_hat = normalize(coords[indices[i].x]);
              double3 r2_hat = normalize(coords[indices[i].y]);
              double3 r3_hat = normalize(coords[indices[i].z]);


              atomicAdd(&out_110_re[(sides[i].z+(out_shape.z*sides[i].y)
                                      +(out_shape.z*out_shape.y*sides[i].x))
                                      +(out_shape.x*out_shape.y*out_shape.z*blockIdx.x)],
                                      -1./(4.*PI)*sqrtf(3./(4.*PI))*dot(r1_hat,r2_hat));
              atomicAdd(&out_111_im[(sides[i].z+(out_shape.z*sides[i].y)
                                      +(out_shape.z*out_shape.y*sides[i].x))
                                      +(out_shape.x*out_shape.y*out_shape.z*blockIdx.x)],
                                      (-3./sqrtf(2.*pow(4.*PI,3.)))*dot(r1_hat,cross(r2_hat,r3_hat)));
              atomicAdd(&out_112_re[(sides[i].z+(out_shape.z*sides[i].y)
                                      +(out_shape.z*out_shape.y*sides[i].x))
                                      +(out_shape.x*out_shape.y*out_shape.z*blockIdx.x)],
                                      sqrtf(27./(2.*pow(4.*PI,3.)))*(dot(r1_hat,r3_hat)*dot(r2_hat,r3_hat)-1./3.*dot(r1_hat,r2_hat)));
              atomicAdd(&out_222_re[(sides[i].z+(out_shape.z*sides[i].y)
                                      +(out_shape.z*out_shape.y*sides[i].x))
                                      +(out_shape.x*out_shape.y*out_shape.z*blockIdx.x)],
                                      -45./(sqrtf(14.*pow(4.*PI,3.))) *(dot(r1_hat,r2_hat)*dot(r1_hat,r3_hat)*dot(r2_hat,r3_hat)
                                      -1./3.*pow(dot(r1_hat,r2_hat),2.)
                                      -1./3.*pow(dot(r1_hat,r3_hat),2.)
                                      -1./3.*pow(dot(r2_hat,r3_hat),2.)+2./9.));
              
          }
        }
        '''
        #Define Kernel Functions for later use
        manip_module = cp.RawModule(code=code_manip)       
        add_module = cp.RawModule(code=code_add)
        P_lambda_module = cp.RawModule(code=P_lambda_code)
        if self.npcf == 3:
            if self.numprimes == 1:
                self.ballct_func = manip_module.get_function('get_ballct_3PCF_single')
                self.sides_func = manip_module.get_function('get_sides_3PCF_single')
                self.N3_and_add = P_lambda_module.get_function('N3_coeff_add_single')
            else:
                self.ballct_func = manip_module.get_function('get_ballct_3PCF')
                self.sides_func = manip_module.get_function('get_sides_3PCF')
                self.N3_and_add = P_lambda_module.get_function('N3_coeff_add')
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
                second_coord_3d_to_pass = cp.array([[],[],[]]).reshape((0,3))
                second_coord_radial_to_pass = cp.array([])
                primes=masterprime+ii
                print(f">> sit on {primes}th galaxy")
                #second_coord_3d,second_coord_radial = self.get_coords(primes[0],eps,tree,galcoords_gpu)
                for n, prime in enumerate(primes):
                    second_coord_3d,second_coord_radial = self.get_coords(prime,eps,tree,galcoords_gpu)
                    second_coord_3d_to_pass = cp.concatenate((second_coord_3d_to_pass,second_coord_3d))
                    second_coord_radial_to_pass = cp.concatenate((second_coord_radial_to_pass,second_coord_radial))
                    coord_lengths_to_pass[n] = len(second_coord_radial)

            self.run_some_gals(ii,second_coord_3d_to_pass,second_coord_radial_to_pass,masterballct,coord_lengths_to_pass)
            #print(benchmark(self.run_1_gal,(ii,eps,tree,galcoords_gpu,masterballct), n_repeat=2,n_warmup=2))
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
        sort = second_coord_radial.argsort()
        second_coord_3d = second_coord_3d[sort]
        second_coord_radial = second_coord_radial[sort]
        #print(type(second_coord_3d))
        return (second_coord_3d,second_coord_radial)
    
    def run_some_gals(self,ii,second_coord_3d,second_coord_radial,masterballct,coord_lengths):
        #print(f">> also sit on {ii+self.ngals/2}th galaxy")
        #mempool = cp.get_default_memory_pool()
        #print(f"Mem tot: {mempool.total_bytes()/1024/1024}")
        #print(f"Mem used: {mempool.used_bytes()/1024/1024}")
        '''
        prime_coord_3d = galcoords_gpu[ii]
        #prime_coords_3d_2 = galcoords_gpu[ii+self.ngals/2]

        # ballct: list of indices of the secondary galaxies of the current primary galaxy
        ballct = tree.query_ball_point(self.galcoords[ii], self.rmax+eps)
        ballct.remove(ii)

        # coordinates of the secondary galaxies 
        second_coord_3d = galcoords_gpu[ballct] - prime_coord_3d   
        second_coord_radial = cp.sqrt(cp.sum(second_coord_3d**2, axis=1))
        '''
        #ballct,second_coord_3d,second_coord_radial = self.get_coords(ii,eps,tree,galcoords_gpu)
        
        if second_coord_3d.shape[0] >= (self.npcf-1): #Excludes primaries with too few secondaries
            
            print(f"Num Secondaries: {second_coord_3d.shape[0]}")
            '''
            if np.any(coord_lengths > self.Nmax):
                print("Warning: Number of secondaries is greater than the number computed")
                self.problems += 1
                return 0
            '''

            #Calculates the number of blocks needed - Should agree with self.Numblocks  
            Numblocks_min = math.ceil(masterballct.shape[0]/1024)+1
            if self.Numblocks < Numblocks_min:
                self.problems += 1
                print("Warning: Number of blocks is fewer than required, computation output will be incorrect")
            print(f"{self.Numblocks=}, {Numblocks_min=}")
            
            # Creates holder array for secondary combinations then
            # populates with the entries from masterballct only if they are
            # possible with the actual number of secondaries
            comb_ballct_arr =  cp.zeros((len(masterballct)*self.numprimes,(self.npcf-1)), dtype=self.ballct_dtype).view(self.ballctvec)
            self.ballct_func((self.Numblocks,), (1024,),(masterballct.view(self.shortvec),masterballct.shape[0],self.numprimes,coord_lengths,second_coord_3d.shape[0],comb_ballct_arr))
            #print(f"Ballct: {comb_ballct_arr.nbytes/1024/1024}")
            #print(f"Master Ballct: {masterballct.nbytes/1024/1024}")
    #            if self.verbose: print(f"  position of {self.npcf-1:1d} secondary galaxies:\n", second_coord_3d_np)


        
            #Gets the radial bin index of the N-1 secondary galaxies
            sides = cp.zeros((len(comb_ballct_arr),(self.npcf-1)), dtype='int16').view(self.shortvec)            
            self.sides_func((self.Numblocks,), (1024,),(second_coord_radial,comb_ballct_arr,len(comb_ballct_arr),self.rmax,self.nbins,sides))
            #print(f"Storage array: {self.zeta_re['110'].nbytes/1024/1024}")
            #print(f"Sides: {sides.nbytes/1024/1024}")
            #print(f"Mem tot: {mempool.total_bytes()/1024/1024}")
            #print(f"Mem used: {mempool.used_bytes()/1024/1024}")

            if sides.shape[0] >= 1: #Excludes primaries with too few secondaries
                second_coord_3d = second_coord_3d.view(self.double3)   
                if self.npcf ==3:
                    #Shape of the output array - used for indexing in function call
                    shape = np.asarray([self.zeta_re[self.lls[0]].shape[1],self.zeta_re[self.lls[0]].shape[2]]).astype(np.int32).view(self.int2)
                    #Calculates coefficients and adds them to zeta
                    self.N3_and_add((self.Numblocks,), (1024,),(second_coord_3d,comb_ballct_arr,sides,
                                                        self.zeta_re['00'],self.zeta_re['11'],self.zeta_re['22'],
                                                        comb_ballct_arr.shape[0],shape))
                elif self.npcf == 4:
                    #Shape of the output array - used for indexing in function call
                    shape = np.asarray([
                        self.zeta_re[self.lls[0]].shape[1],
                        self.zeta_re[self.lls[0]].shape[2],
                        self.zeta_re[self.lls[0]].shape[3]]).astype(np.int32).view(self.int3)
                    #Calculates coefficients and adds them to zeta arrray
                    self.N4_and_add((self.Numblocks,), (1024,),(second_coord_3d,comb_ballct_arr,sides,
                                                        self.zeta_re['110'],self.zeta_im['111'],self.zeta_re['112'],self.zeta_re['222'],
                                                        comb_ballct_arr.shape[0],shape))
                    
                #print(f"Mem tot: {mempool.total_bytes()/1024/1024}")
                #print(f"Mem used: {mempool.used_bytes()/1024/1024}")
        #                if self.verbose: print("    order", il, "coefficients",new_zeta)
        #         self.zeta = numpy.average(numpy.array(list(self.zeta.values())),axis=0).real
            #print(cp.any(self.zeta_im['00'] != 0))


if __name__ == "__main__":

    start_time = time.time()

    npcf = 3
    ngals = 5000
    nbins = 10
    lbox = 20
    rmax = 20
    Nmax = 10000
    numprimes = 1
    #Numblocks = 306000
    # lls_5pcf = ['11(0)11', '21(1)12']
    verbose=False
    
    zetas = calc_NPCF(npcf, ngals, nbins, lbox, rmax, Nmax,numprimes, lls=None, verbose=verbose)
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
