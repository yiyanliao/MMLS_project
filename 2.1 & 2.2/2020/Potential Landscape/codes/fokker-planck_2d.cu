////----------------------------------------
// two-dimensional Fokker-Planck equation
// 
//----------------------------------------
// parameters of simulations are in "input_FP.par"
//
// compile:
// nvcc fokker-planck_2d.cu -lcuda -lcufft -lcublas -O3 -arch sm_20 -o fp2d
//
//  produces output files "n_#####.dat" with (i, j, n, S) columns
//  
// 
//
// uses complex.h
// and semi-implicit algorithm


#include<iostream>
#include<fstream>
#include<cstring>
#include<stdlib.h>
#include<math.h>


#include <iomanip>
#include <cmath>
#include <queue>

/* we need these includes for CUDA's random number stuff */
#include <unistd.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>


//#include "fileutils.h"
//#include "stringutils.h"

// #include<complex.h>
// #include<fftw3.h>

#include <cuda.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// #include <cutil_inline.h>
// #include "reduction_kernel.cu"

#define N 256// was 256
#define NX  (N*N)

using namespace std;

//#define DBLPREC

#ifdef DBLPREC
#define MAXT 512     //was 512
typedef double REAL;
#else
#define MAXT 128     //was 1024
typedef float REAL;
#endif


typedef struct {
    REAL re;
    REAL im;
} COMPLEX;

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
  /* we have to initialize the state */
    if(idx<NX)
    {
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              idx, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[idx]);
    }
}



typedef struct  {
        //  parameters
        REAL k1dt, k2dt, k3dt, k4dt, k5dt, k6dt, KM1, KM2, KM3, KM4, Df, alpha, beta, St;
} sysvar;

//===========================================================================
double rn()
{
    return drand48();
}
//===========================================================================
//CUDA Kernels

//===========================================================================

__global__ void r2c(COMPLEX *cS, REAL *S, REAL *S1 )
{
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<NX)
    {
        cS[idx].re=S[idx];
        cS[idx].im=S1[idx];
    }
};

__global__ void r2c1(COMPLEX *cS, REAL *S )
{
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<NX)
    {
        cS[idx].re=S[idx];
        cS[idx].im=0.;
    }
};
__global__ void c2r(COMPLEX *cS, REAL *S, REAL *S1 )
{
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<NX)
    {
        S[idx]=cS[idx].re;
        S1[idx]=cS[idx].im;
    }
};


/* this GPU kernel takes an array of states, and an array of ints, and
 * puts a random int into each */
__global__ void randoms(curandState_t* states, REAL * numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<NX)
    {
        numbers[idx] = curand_uniform(&states[idx]);
    }
}


//---------------------------------------------------------------------------
//update P 
//---------------------------------------------------------------------------


__global__  void flux(REAL *H, REAL *S, REAL *P, REAL *P1, sysvar* cu_Vars)
{
	REAL h, s, h2, s2, h4, s4;
	REAL Ahp, Asp, hp, sp, hp2, sp2, hp4, sp4;
	REAL Ahm, Asm, hm, sm, hm2, sm2, hm4, sm4;
	REAL Dhm, Dsm, Dhp, Dsp;
	int i,j;

	// read parameters
	REAL k1dt = cu_Vars[0].k1dt;
	REAL k2dt = cu_Vars[0].k2dt;
	REAL k3dt = cu_Vars[0].k3dt;
	REAL k4dt = cu_Vars[0].k4dt;
	REAL k5dt = cu_Vars[0].k5dt;
	REAL k6dt = cu_Vars[0].k6dt;
	REAL KM1 = cu_Vars[0].KM1;
	REAL KM2 = cu_Vars[0].KM2;
	REAL KM3 = cu_Vars[0].KM3;
	REAL KM4 = cu_Vars[0].KM4;
	REAL alpha = cu_Vars[0].alpha;
	REAL beta = cu_Vars[0].beta;
	REAL St = cu_Vars[0].St;
	REAL Df = cu_Vars[0].Df;

	int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

	REAL factor=St/15000;
	if(idx>=0 && idx<NX)
	{
		h=H[idx];
		s=factor*S[idx];
		h2=h*h; s2=s*s;
		h4=h2*h2; s4=s2*s2;

		i=(int)(idx/N);
		j=idx-i*N;
		hp=H[j+N*(i+1)];
		hm=H[j+N*(i-1)];
		hp2=hp*hp; hp4=hp2*hp2;
	        hm2=hm*hm; hm4=hm2*hm2;
                sp=factor*S[j+1+N*i];
                sm=factor*S[j-1+N*i];
	        sp2=sp*sp; sp4=sp2*sp2;
	        sm2=sm*sm; sm4=sm2*sm2;
		if(i>0&&i<N-1){
			Ahp=P[j+N*(i+1)]*(k1dt*((1-alpha)*s2+KM1*KM1)/(s2+KM1*KM1)*hp4/(hp4+KM2*KM2*KM2*KM2)+k2dt-k3dt*hp);
			Ahm=P[j+N*(i-1)]*(k1dt*((1-alpha)*s2+KM1*KM1)/(s2+KM1*KM1)*hm4/(hm4+KM2*KM2*KM2*KM2)+k2dt-k3dt*hm);
			Dhp=P[j+N*(i+1)]-P[j+N*i];
			Dhm=P[j+N*(i-1)]-P[j+N*i];
		}
		else if (i==0){
                        Ahp=P[j+N*(i+1)]*(k1dt*((1-alpha)*s2+KM1*KM1)/(s2+KM1*KM1)*hp4/(hp4+KM2*KM2*KM2*KM2)+k2dt-k3dt*hp);
			Ahm=-P[j+N*(i)]*(k1dt*((1-alpha)*s2+KM1*KM1)/(s2+KM1*KM1)*h4/(h4+KM2*KM2*KM2*KM2)+k2dt-k3dt*h);
			Dhp=P[j+N*(i+1)]-P[j+N*i];
			Dhm=0;
                }
		else if (i==N-1){
                        Ahp=-P[j+N*(i)]*(k1dt*((1-alpha)*s2+KM1*KM1)/(s2+KM1*KM1)*h4/(h4+KM2*KM2*KM2*KM2)+k2dt-k3dt*h);
                        Ahm=P[j+N*(i-1)]*(k1dt*((1-alpha)*s2+KM1*KM1)/(s2+KM1*KM1)*hm4/(hm4+KM2*KM2*KM2*KM2)+k2dt-k3dt*hm);
			Dhp=0;
			Dhm=P[j+N*(i-1)]-P[j+N*i];
                }
		
                if(j>0&&j<N-1){
                        Asp=P[j+1+N*i]*(k4dt*((1-beta)*h2+KM3*KM3)/(h2+KM3*KM3)*sp4/(sp4+KM4*KM4*KM4*KM4)*(St-sp)+k5dt-k6dt*sp);
                        Asm=P[j-1+N*i]*(k4dt*((1-beta)*h2+KM3*KM3)/(h2+KM3*KM3)*sm4/(sm4+KM4*KM4*KM4*KM4)*(St-sm)+k5dt-k6dt*sm);
			Dsp=P[j+1+N*i]-P[j+N*i];
			Dsm=P[j-1+N*i]-P[j+N*i];
	        }
		else if (j==0) {
                        Asp=P[j+1+N*i]*(k4dt*((1-beta)*h2+KM3*KM3)/(h2+KM3*KM3)*sp4/(sp4+KM4*KM4*KM4*KM4)*(St-sp)+k5dt-k6dt*sp);
                        Asm=-P[j+N*i]*(k4dt*((1-beta)*h2+KM3*KM3)/(h2+KM3*KM3)*s4/(s4+KM4*KM4*KM4*KM4)*(St-s)+k5dt-k6dt*s);
			Dsp=P[j+1+N*i]-P[j+N*i];
			Dsm=0;
                }
		else if (j==N-1) {
                        Asp=-P[j+N*i]*(k4dt*((1-beta)*h2+KM3*KM3)/(h2+KM3*KM3)*s4/(s4+KM4*KM4*KM4*KM4)*(St-s)+k5dt-k6dt*s);
                        Asm=P[j-1+N*i]*(k4dt*((1-beta)*h2+KM3*KM3)/(h2+KM3*KM3)*sm4/(sm4+KM4*KM4*KM4*KM4)*(St-sm)+k5dt-k6dt*sm);
			Dsp=0;
			Dsm=P[j-1+N*i]-P[j+N*i];
                }

           //P1[idx]=P[idx]-Ahp+Ahm-Asp+Asm+Df*(Dhp+Dhm+Dsp+Dsm);
           P1[idx]=P[idx]-Ahp+Ahm-Asp+Asm+Df*(Dhp+Dhm)+0.01*Df*(Dsp+Dsm);
	   if(P1[idx]<0) P1[idx]=0;
    }
};

__global__  void combine(COMPLEX *GNdFx, COMPLEX *GNdFy, COMPLEX *GNdF)
{
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<NX)
    {
        GNdF[idx].re=GNdFx[idx].re+GNdFy[idx].re;
        GNdF[idx].im=GNdFx[idx].im+GNdFy[idx].im;
    }
};


__global__ void copy_P(REAL *n, REAL *na)
{

    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

    if(idx<NX)
    {
            na[idx]=n[idx];
    }
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    dim3 dGRID,dBLOCK;
    int GPUID=0;

    REAL *H,*GH;
    REAL *S,*GS;
    REAL *R,*GR;
    REAL *P, *P1, *GP, *GP1, *GCT1, *GCT2; 
    COMPLEX  *GcP, *GcFx, *GcFy, *GNcP, *GNcFx, *GNcFy, *GNdF, *GNdFx, *GNdFy, *GCT, *complexdummy;
    REAL *cor1,*Gcor1,*realdummy;
    COMPLEX *cor3x,*Gcor3x, *cor3y, *Gcor3y;
    cufftHandle Gfftplan;
//     cublass stuff------------------------------------------------

    cublasHandle_t handle;
//-----------------------------------------------------------------

    int memNXc,memNXr;

    REAL k1,k2,k3,k4,k5,k6,KM1,KM2,KM3,KM4,alpha,beta,St,D,Df;
    REAL totP,minP,maxP;
    REAL LX, dt;
//
    double qx[N],qy[N],qsq;
    string ext;  // for file I/O
    string file;
    char datname[200],filename[200],comd[200];
    int steps,interval,zahl; 

    REAL dx;	//dx=L/N
    double dkx,scale=N*N;
    int i,j,k,k0;

    double tim=0.,timestart=0.0;

    string s;


    cor1 = new REAL[NX];
    cor3x = new COMPLEX[NX];
    cor3y = new COMPLEX[NX];
    S = new REAL[NX];
    H = new REAL[NX];
    P = new REAL[NX];
    R = new REAL[NX];
    P1 = new REAL[NX];


    realdummy = new REAL[NX];    
    complexdummy = new COMPLEX[NX];    

    cudaThreadExit();
    cudaSetDevice(GPUID);
    dBLOCK=dim3(MAXT,1);
    i=512; //x blocks, limited to 256^2 !!!
    k=i*dBLOCK.x;
    j=(NX+k-1)/k; //y blocks
// printf("i=%d j=%d k=%d dBLOCK.x=%d\n",i,j,k,dBLOCK.x);
    dGRID=dim3(i,j);


    // Create CUDA FFT plan
#ifdef DBLPREC
    cufftPlan2d(&Gfftplan, N, N, CUFFT_Z2Z) ;
    printf("double precision code\n");
#else
    cufftPlan2d(&Gfftplan, N, N, CUFFT_C2C) ;
#endif

    cublasCreate(&handle);



    memNXc=NX*sizeof(COMPLEX);
    memNXr=NX*sizeof(REAL);

    //complex arrays on GPU
    cudaMalloc((void**)&GcP, memNXc);
    cudaMalloc((void**)&GcFx, memNXc);
    cudaMalloc((void**)&GcFy, memNXc);
    cudaMalloc((void**)&GNcP, memNXc);
    cudaMalloc((void**)&GNcFx, memNXc);
    cudaMalloc((void**)&GNcFy, memNXc);
    cudaMalloc((void**)&GNdF, memNXc);
    cudaMalloc((void**)&GNdFx, memNXc);
    cudaMalloc((void**)&GNdFy, memNXc);
    cudaMalloc((void**)&GCT, memNXc);
    cudaMalloc((void**)&Gcor3x, memNXc);
    cudaMalloc((void**)&Gcor3y, memNXc);

    //real arrays on GPU
    cudaMalloc((void**)&GP,memNXr);
    cudaMalloc((void**)&GS,memNXr);
    cudaMalloc((void**)&GH,memNXr);
    cudaMalloc((void**)&GR,memNXr);
    cudaMalloc((void**)&Gcor1, memNXr);
    cudaMalloc((void**)&GCT1,memNXr);
    cudaMalloc((void**)&GCT2,memNXr);
    cudaMalloc((void**)&GP1,memNXr);


        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////

        // parameter input

        const int iNTRASH=256;
        char trashbuffer[iNTRASH];

        ifstream fin("input_FP.par");

        if (!fin.good()) {
                cerr << "Cannot find input_FP.par" << endl;
                return 1;
        } // if

        const int iINPUTSIZE = 17;
        REAL dFIP [iINPUTSIZE];

        int iFileLine = 0;
        while ((iFileLine < iINPUTSIZE) && (fin.good())) {
                fin >> dFIP[iFileLine];
                fin.getline(trashbuffer, iNTRASH);
                //cout << dFIP[iFileLine] << "\t" << trashbuffer <<
                //endl;
                iFileLine++;
        } // while


        k1=dFIP[0];
        k2=dFIP[1];
        k3=dFIP[2];
        k4=dFIP[3];
        k5=dFIP[4];
        k6=dFIP[5];
        KM1=dFIP[6];
        KM2=dFIP[7];
        KM3=dFIP[8];
        KM4=dFIP[9];
        alpha=dFIP[10];
        beta=dFIP[11];
        St=dFIP[12];
        D=dFIP[13];
	dt=dFIP[14];
        steps=dFIP[15];
        interval=dFIP[16];


        fin.getline(trashbuffer, iNTRASH);

        fin.close();

 //	LX=2*NX; //system size
 	LX=15000; //system size
        dx=LX/N;
        dkx=2.*M_PI/LX;
        REAL coef=1./2./dx;
	Df=D*dt/dx/dx;


        cerr << "k1: \t" << k1 << "\n";
        cerr << "k2: \t" << k2 << "\n";
        cerr << "k3: \t" << k3 << "\n";
        cerr << "k4: \t" << k4 << "\n";
        cerr << "k5: \t" << k5 << "\n";
        cerr << "k6: \t" << k6 << "\n";
        cerr << "KM1: \t" << KM1 << "\n";
        cerr << "KM2: \t" << KM2 << "\n";
        cerr << "KM3: \t" << KM3 << "\n";
        cerr << "KM4: \t" << KM4 << "\n";
        cerr << "alpha: \t" <<alpha << "\n";
        cerr << "beta: \t" <<beta << "\n";
        cerr << "St: \t" <<St << "\n";
        cerr << "D: \t" << D << "\n";
        cerr << "LX: \t" << LX << "\n";
        cerr << "dt: \t" << dt << "\n";
        cerr << "steps: \t" << steps << "\n";
        cerr << "interval: \t" << interval << "\n";



        k1=coef*k1;
        k2=coef*k2;
        k3=coef*k3;
        k4=coef*k4;
        k5=coef*k5;
        k6=coef*k6;


  //  sysvar sv_Vars [NX];
    sysvar sv_Vars[1];
                // parameters
                sv_Vars[0].k1dt = k1*dt;
                sv_Vars[0].k2dt = k2*dt;
                sv_Vars[0].k3dt = k3*dt;
                sv_Vars[0].k4dt = k4*dt;
                sv_Vars[0].k5dt = k5*dt;
                sv_Vars[0].k6dt = k6*dt;
                sv_Vars[0].KM1 = KM1;
                sv_Vars[0].KM2 = KM2;
                sv_Vars[0].KM3 = KM3;
                sv_Vars[0].KM4 = KM4;
                sv_Vars[0].alpha = alpha;
                sv_Vars[0].beta = beta;
                sv_Vars[0].St = St;
                sv_Vars[0].Df = Df;



        // allocate memory on the device 
        sysvar* cu_Vars;
        size_t size_Vars = sizeof(sv_Vars);
        cudaMalloc((void**)&cu_Vars, size_Vars);


        // copy to the device
        cudaMemcpy(cu_Vars, sv_Vars, size_Vars, cudaMemcpyHostToDevice);
//        cudaMemcpy(sv_Vars, cu_Vars, size_Vars, cudaMemcpyDeviceToHost);
//  cerr << "gamma1dt: \t" << sv_Vars[0].gamma1dt << "\n";

  /* CUDA's random number library uses curandState_t to keep track of
 * the seed value we will store a random state for every thread  */
  curandState_t* states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, NX * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  //init<<<dGRID, dBLOCK>>>(time(0), states);
  init<<<dGRID, dBLOCK>>>(time(0), states);

  /* allocate an array of floats on the CPU and GPU */
//  float cpu_nums[NX];
//  float cpu_nums1[NX]; 
  REAL *gpu_nums1;

  cudaMalloc((void**) &gpu_nums1, NX * sizeof(REAL));

//  randoms <<<dGRID, dBLOCK>>>(states, gpu_nums1);
//            cudaMemcpy(cpu_nums, gpu_nums1, NX * sizeof(float), cudaMemcpyDeviceToHost);
//	for(i=0;i<200;i++)
//	for(i=NX-20;i<NX;i++)
//	   printf("test: %f\n",cpu_nums[i]);

    //new simulation

        for(i=0; i<N; i++)
       	for(j=0; j<N; j++)
	{
        	H[j+N*i]=dx*i;
        	S[j+N*i]=dx*j;
        }

// initial conditions mark

int NEW;
if(argc<2)
    NEW=1;
else
    NEW=0;

if(NEW==1){
        k0=0;
        for(i=0; i<N; i++)
          for(i=0; i<N; i++){
        	P[j+N*i]=0; 
        	realdummy[j+N*i] = 0.;
        	complexdummy[j+N*i].re = 0.;
        	complexdummy[j+N*i].im = 0.;
	}


 	REAL W=2000;
	totP=0;
        for(i=0; i<N; i++)
        	for(j=0; j<N; j++)
            {
		REAL arg1=dx*(i-1.*N/4.)/W;
		REAL arg2=dx*(j-N/4.)/W;
		P[j+N*i]=exp(-(arg1*arg1+arg2*arg2));

		totP+=P[j+N*i] ;
		realdummy[j+N*i] = 0.;
		complexdummy[j+N*i].re = 0.;
		complexdummy[j+N*i].im = 0.;
            }

        cout << "initial totP:" <<  "\t" << totP  <<endl;
}
else
{
ifstream fin1("last_snapshot.dat");

        if (!fin1.good()) {
                cerr << "Cannot find last_snapshot.dat" << endl;
                return 1;
        } // if

                i = 0;
                fin1 >> k0;
                fin1.getline(trashbuffer, iNTRASH);
        while ((i < NX) && (fin.good())) {
                fin1 >> P[i];
                fin1.getline(trashbuffer, iNTRASH);
                i++;
                totP+=P[i] ;
        } // while
        fin1.close();
        cout << "initial totP:" <<  "\t" << totP  <<endl;

}



        //

    //output params in file
 //   file=ext+"/params.dat";  		//add path to file
    file="params.dat";  		//add path to file
    strcpy(filename,file.c_str());		//copy to cstring
    ofstream outsp;
    outsp.open (filename,ofstream::out );	//open file
    {
        outsp << "parameters" << endl
              << "----------" << endl
              << "grid size:\t" << N << endl
              << "steps      :\t" << steps << endl
              << "L       :\t" << LX << endl
              << "dt      :\t" << dt << endl
              << "k1     :\t" << k1 << endl
              << "k2     :\t" << k2 << endl
              << "k3     :\t" << k3 << endl
              << "k4     :\t" << k4 << endl
              << "k5     :\t" << k5 << endl
              << "k6     :\t" << k6 << endl
              << "KM1     :\t" << KM1 << endl
              << "KM2     :\t" << KM2 << endl
              << "KM3     :\t" << KM3 << endl
              << "KM4     :\t" << KM4 << endl
              << "alpha   :\t" << alpha << endl
              << "beta     :\t" << beta << endl
              << "St   :\t" << St << endl
              << "D     :\t" << D << endl;
    }
    outsp.close();

    //calculate q's for fourier operations
    for(i=0; i<=N/2; i++)
    {
        qx[i]=dkx*i; //dk/2*i;
        qy[i]=dkx*i; //dk/2*i;
    }
    for(i=1; i<N/2; i++)
    {
        qx[N-i]=-dkx*i; //-dk/2*i;
        qy[N-i]=-dkx*i; //-dk/2*i;
    }
  //      qx[N/2]=0;  (only for first derivatives)

/*
    //cout << endl;
    cout << "----------------------------" << endl;
    cout << "qmax   : " << qx[N/2]  << endl;
    cout << "Deltaq : " << qx[1] << "\t" << dk << endl;
    cout << "sys-L  : " << L << "\t" << N*dx << endl;
    cout << "dx     : " << dx << endl;
    cout << "----------------------------" << endl;
*/

    // cor matrix
    for(i=0; i<N; i++)
        for (j=0; j<N; j++)
        {
            qsq=qx[i]*qx[i]+qy[j]*qy[j];
            cor1[j+N*i]=exp(-dt*D*qsq)/scale;
            cor3x[j+N*i].re=0;
            cor3x[j+N*i].im=qx[i]*exp(-0.1*D*dt*qsq)/scale;
            cor3y[j+N*i].re=0;
            cor3y[j+N*i].im=qy[j]*exp(-0.1*D*dt*qsq)/scale;
        }


    cudaMemcpy(GS, S,  memNXr, cudaMemcpyHostToDevice);
    cudaMemcpy(GH, H,  memNXr, cudaMemcpyHostToDevice);
    cudaMemcpy(GP, P,  memNXr, cudaMemcpyHostToDevice);
    cudaMemcpy(Gcor1, cor1,  memNXr, cudaMemcpyHostToDevice);
    cudaMemcpy(Gcor3x, cor3x,  memNXc, cudaMemcpyHostToDevice);
    cudaMemcpy(Gcor3y, cor3y,  memNXc, cudaMemcpyHostToDevice);

//initialize all other GPU Arrays with dummy zeros - seems to be necessary for older
//Graphic Cards like GeForce GTX 285

//    cudaMemcpy(GNcS, complexdummy, memNXc, cudaMemcpyHostToDevice);
//    cudaMemcpy(GNdS, complexdummy, memNXc, cudaMemcpyHostToDevice);


int iout=0;
//timestep
//--------------------------------------------------------------------------------------
    for(k=k0; k<steps+1; k++)
    {

        tim+=dt;



// plot output
     //   if(k%interval==0 && k!=0)
        if(k%interval==0 )
        {
            zahl=int(timestart)+int(k/interval);

            cout << "time=" << k*dt << endl;

//---copy data from GPU to CPU -----------------
//            cudaMemcpy(S,   GS,   memNXr, cudaMemcpyDeviceToHost);
//            cudaMemcpy(H,   GH,   memNXr, cudaMemcpyDeviceToHost);
            cudaMemcpy(P, GP, memNXr, cudaMemcpyDeviceToHost);
            cudaMemcpy(R, GR, memNXr, cudaMemcpyDeviceToHost);


            //output P  in file
            sprintf(datname,"n_%5.5d.dat",zahl);
//            file=ext+"/"+datname;  			//add path to file
            file=datname;  		
            strcpy(filename,file.c_str());		//copy to cstring

            ofstream outsr;
            outsr.open (filename, ofstream::out );
            for(i=0; i<NX; i++)
            {
                    outsr // << i*dx 
                          << "\t" << P[i]
                      //    << "\t" << R[i]
                          << endl;
            }
            outsr.close();

            system("rm -f last_snapshot.dat");
            file="last_snapshot.dat";
            strcpy(filename,file.c_str());              //copy to cstring
            outsr.open (filename, ofstream::out );
                    outsr << k << endl;
/*
            for(i=0; i<NX; i++)
            {
                    outsr << P[i] << endl;
            }
*/
            outsr.close();

            sprintf(comd,"cat %s >> last_snapshot.dat",datname);
            system(comd);

        totP=0;
	minP=1000;
	maxP=-1000;
            for(i=0; i<N; i++)
            for(j=0; j<N; j++)
            {
                totP+=P[j+N*i] ;
		if (P[j+N*i]>maxP)maxP=P[j+N*i];
		if (P[j+N*i]<minP)minP=P[j+N*i];
           }
        cout << "current totP: " <<  "\t" << totP  << "   minP: " << minP << "   maxP: " << maxP <<endl;

	iout++;
        }// end output



////////////////////////////////////////////////////////////////////


// compute diffusion and drift terms 
   int SH=0;
   REAL totPb=0;
   REAL totPa=0;
   if(SH==1){
        cudaMemcpy(P, GP, memNXr, cudaMemcpyDeviceToHost);
            for(i=0; i<NX; i++)
            {
                totPb+=P[i] ;
           }
        // cout << "totP before:" <<  "\t" << totP  <<endl;
    }
        flux<<<dGRID, dBLOCK>>>(GH, GS, GP, GP1, cu_Vars); //compute d/dH(Ax(H,S)*P) and d/dS(Ay(H,S)*P)
	copy_P<<<dGRID, dBLOCK>>>(GP1, GP); //copy GP1 into GP

   if(SH==1){
        cudaMemcpy(P1, GP1, memNXr, cudaMemcpyDeviceToHost);
            for(i=0; i<NX; i++)
            {
                totPa+=P1[i] ;
           }
        //if(k%interval==0 ) 
        if(k==0 ) 
          cout << "totP diff=" << totPa-totPb <<endl;
	}

///////////////////////////////////////////////////////////////////////////////////////////////

    }// end timestep

        totP=0;
	minP=1000;
	maxP=-1000;
            for(i=0; i<NX; i++)
            {
                totP+=P[i] ;
		if (P[i]>maxP)maxP=P[i];
		if (P[i]<minP)minP=P[i];
           }
        cout << "current totP:" <<  "\t" << totP  << "minP:" << minP << "maxP:" << maxP <<endl;
/*
           ofstream outsr1;
           outsr1.open ("density.d", ofstream::out | ofstream::app );
           outsr1  << St << "\t"
                          << "\t" << totP
                          << endl;
            outsr1.close();
*/
          
        // free up the allocated memory on the device

    delete[] P;
    delete[] cor1;
    delete[] realdummy;
    delete[] complexdummy;

    cudaFree(GP);
    cudaFree(GcP);
    cudaFree(GcFx);
    cudaFree(GcFy);
    cudaFree(GNcP);
    cudaFree(GNcFx);
    cudaFree(GNcFy);
    cudaFree(GNdF);
    cudaFree(GNdFx);
    cudaFree(GNdFy);
    cudaFree(Gcor1);
    cudaFree(Gcor3x);
    cudaFree(Gcor3y);

    cufftDestroy(Gfftplan);
    cublasDestroy(handle);

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  cudaFree(gpu_nums1);

   cudaError_t errcode = cudaGetLastError();
    return 0;
}
