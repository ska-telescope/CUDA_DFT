
// Copyright 2019 Seth Hall, Adam Campbell, Andrew Ensor
// High Performance Computing Research Laboratory, 
// Auckland University of Technology (AUT)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DIRECT_FOURIER_TRANSFORM_H_
#define DIRECT_FOURIER_TRANSFORM_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// Pi (double precision)
#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

// Speed of light
#ifndef C
	#define C 299792458.0
#endif

// Flip between single/double (ONLY DOUBLE PRECISION FULLY OPTIMIZED AT THIS STAGE)
#ifndef SINGLE_PRECISION
	#define SINGLE_PRECISION 0
#endif

// Define global precision
#ifndef PRECISION
	#if SINGLE_PRECISION
		#define PRECISION float
	#else
		#define PRECISION double
	#endif
#endif

#ifndef PRECISION2
	#if SINGLE_PRECISION
		#define PRECISION2 float2
	#else
		#define PRECISION2 double2
	#endif
#endif

#ifndef PRECISION3
	#if SINGLE_PRECISION
		#define PRECISION3 float3
	#else
		#define PRECISION3 double3
	#endif
#endif

#ifndef PRECISION4
	#if SINGLE_PRECISION
		#define PRECISION4 float4
	#else
		#define PRECISION4 double4
	#endif
#endif

#if SINGLE_PRECISION
	#define MAKE_PRECISION2(x,y) (make_float2(x,y))
#else
	#define MAKE_PRECISION2(x,y) (make_double2(x,y))
#endif

#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)

typedef struct Config {
	int num_visibilities;
	int num_sources;
	const char *source_file;
	const char *vis_src_file;
	const char *vis_dest_file;
	bool synthetic_sources;
	bool synthetic_visibilities;
	bool gaussian_distribution_sources;
	bool force_zero_w_term;
	double min_u;
	double max_u;
	double min_v;
	double max_v;
	double min_w;
	double max_w;
	double grid_size;
	double cell_size;
	double uv_scale;
	double frequency_hz;
	int gpu_num_blocks;
	int gpu_num_threads;
	bool enable_messages;
} Config;

typedef struct Complex {
	double real;
	double imaginary;
} Complex;

typedef struct Source {
	double l;
	double m;
	double intensity;
} Source;

typedef struct Visibility {
	double u;
	double v;
	double w;
} Visibility;

void init_config (Config *config);

void load_sources(Config *config, Source **sources);

void load_visibilities(Config *config, Visibility **visibilities, Complex **vis_intensity);

void extract_visibilities(Config *config, Source *sources, Visibility *visibilities,
	Complex *vis_intensity, int num_visibilities);

void save_visibilities(Config *config, Visibility *visibilities, Complex *vis_intensity);

PRECISION random_in_range(PRECISION min, PRECISION max);

PRECISION generate_sample_normal(void);

__global__ void direct_fourier_transform(const PRECISION3 *visibility, PRECISION2 *vis_intensity,
	const int vis_count, const PRECISION3 *sources, const int source_count);

static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

void unit_test_init_config(Config *config);

double unit_test_generate_approximate_visibilities(void);

#endif /* DIRECT_FOURIER_TRANSFORM_H_ */

#ifdef __cplusplus
}
#endif