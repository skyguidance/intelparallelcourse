#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <mkl.h>
#include "distribution.h"

int diffusion(const int n_particles,
              const int n_steps, 
              const float x_threshold, 
              const float alpha,
              VSLStreamStatePtr rnStream);


// DO NOT MODIFY THIS FUNCTION //
//unoptimized reference function
int ref_diffusion(const int n_particles,
                  const int n_steps, 
                  const float x_threshold, 
                  const float alpha,
                  VSLStreamStatePtr rnStream) {
  int n_escaped=0;
  for (int i = 0; i < n_particles; i++) {
    float x = 0.0f;
    for (int j = 0; j < n_steps; j++) {
      float rn;
      vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, 1, &rn, -1.0, 1.0);
      x += delta_max*sinf(alpha*rn)*expf(-rn*rn);
    }
    if (x > x_threshold) n_escaped++;
  }
  return n_escaped;
}

int main(int argc, char** argv) {
  float alpha = 1.0f; 
  float x_threshold = 3.0f;
  if(argc>1) {
    alpha = atof(argv[1]);
  }
  if(argc>2) {
    x_threshold = atof(argv[2]);
  }
  const int n_particles = 1<<17;
  const int n_steps = 500;

  
  VSLStreamStatePtr rnStream;
  //initialize random stream
  vslNewStream( &rnStream, VSL_BRNG_MT19937, 0);
  //compute refernce data
  const int ref_escaped = ref_diffusion(n_particles, n_steps, x_threshold, alpha, rnStream);

  const int n_trials = 10;
  const int skip_trials = 2;
  double tsum = 0.0;
  bool err = false;
  
  //compute diffusion data using function defined in worker.cc and get the timing
  const double t0 = omp_get_wtime();
  int n_escaped = diffusion(n_particles, n_steps, x_threshold, alpha, rnStream);
  const double t1 = omp_get_wtime();
  
  //verify the filter data with refernce data
  if(n_escaped - ref_escaped > 5*sqrt(ref_escaped)) { 
    printf("Error: n_escaped %d, while reference is %d\n", n_escaped, ref_escaped);
  } else {
    // Printing verification and performance
    printf("%d\t(ref: %d)\t%f\n",  n_escaped, ref_escaped, t1-t0);
  }
  
  
}