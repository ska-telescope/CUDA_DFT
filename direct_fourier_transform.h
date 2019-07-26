
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DIRECT_FOURIER_TRANSFORM_H_
#define DIRECT_FOURIER_TRANSFORM_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// Speed of light
#ifndef C
	#define C 299792458.0
#endif

#ifndef SINGLE_PRECISION
	#define SINGLE_PRECISION 1
#endif

#if SINGLE_PRECISION
	#define PRECISION float
	#define PRECISION2 float2
	#define PRECISION3 float3
	#define PRECISION4 float4
	#define PI ((float) 3.141592654)
#else
	#define PRECISION double
	#define PRECISION2 double2
	#define PRECISION3 double3
	#define PRECISION4 double4
	#define PI ((double) 3.1415926535897931)
#endif

#if SINGLE_PRECISION
	#define SIN(x) sinf(x)
	#define COS(x) cosf(x)
	#define SINCOS(x, y, z) sincosf(x, y, z)
	#define ABS(x) fabs(x)
	#define SQRT(x) sqrtf(x)
	#define ROUND(x) roundf(x)
	#define CEIL(x) ceilf(x)
	#define LOG(x) logf(x)
	#define POW(x, y) powf(x, y)
	#define MAKE_PRECISION2(x,y) make_float2(x,y)
	#define MAKE_PRECISION3(x,y,z) make_float3(x,y,z)
	#define MAKE_PRECISION4(x,y,z,w) make_float4(x,y,z,w)
#else
	#define SIN(x) sin(x)
	#define COS(x) cos(x)
	#define SINCOS(x, y, z) sincos(x, y, z)
	#define ABS(x) abs(x)
	#define SQRT(x) sqrt(x)
	#define ROUND(x) round(x)
	#define CEIL(x) ceil(x)
	#define LOG(x) log(x)
	#define POW(x, y) pow(x, y)
	#define MAKE_PRECISION2(x,y) make_double2(x,y)
	#define MAKE_PRECISION3(x,y,z) make_double3(x,y,z)
	#define MAKE_PRECISION4(x,y,z,w) make_double4(x,y,z,w)
#endif

#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)

typedef struct Config {
	int num_visibilities;
	int num_predicted_vis;
	int num_sources;
	const char *source_file;
	const char *vis_src_file;
	const char *vis_dest_file;
	bool synthetic_sources;
	bool synthetic_visibilities;
	bool gaussian_distribution_sources;
	bool force_zero_w_term;
	bool enable_right_ascension;
	PRECISION min_u;
	PRECISION max_u;
	PRECISION min_v;
	PRECISION max_v;
	PRECISION min_w;
	PRECISION max_w;
	double grid_size;
	double cell_size;
	double uv_scale;
	double frequency_hz;
	int num_frequencies;
	PRECISION frac_fine_frequency;
	int gpu_max_threads_per_block;
	bool enable_messages;
} Config;

typedef struct Complex {
	PRECISION real;
	PRECISION imaginary;
} Complex;

typedef struct Source {
	PRECISION l;
	PRECISION m;
	PRECISION intensity;
} Source;

typedef struct Visibility {
	PRECISION u;
	PRECISION v;
	PRECISION w;
} Visibility;

void init_config (Config *config);

void load_sources(Config *config, Source **sources);

void load_visibilities(Config *config, Visibility **visibilities, Visibility **predicted_vis,
	Complex **vis_intensity);

void extract_visibilities(Config *config, Source *sources, Visibility *vis_input_uvw,
	Visibility *vis_predicted, Complex *vis_intensity);

void save_visibilities(Config *config, Visibility *predicted, Complex *intensities);

PRECISION random_in_range(PRECISION min, PRECISION max);

PRECISION generate_sample_normal(void);

__device__ PRECISION2 complex_mult(const PRECISION2 z1, const PRECISION2 z2);

__global__ void direct_fourier_transform(const PRECISION3 *d_vis_uvw, PRECISION3 *d_predicted_vis, 
	PRECISION2 *d_intensities, const PRECISION frac_fine_frequency, const int num_vis, const int num_predicted_vis, 
	const PRECISION3 *sources, const int num_sources, const int num_frequencies);

static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

void unit_test_init_config(Config *config);

PRECISION unit_test_generate_approximate_visibilities(void);

#endif /* DIRECT_FOURIER_TRANSFORM_H_ */

#ifdef __cplusplus
}
#endif