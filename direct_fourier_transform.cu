
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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <numeric>

#include "direct_fourier_transform.h"

//IMPORTANT: Modify configuration for target GPU and DFT
void init_config(Config *config)
{
	/// Number of sources to process
	config->num_sources = 1;

	// Number of visibilities per source
	config->num_visibilities = 1;

	// Disregard visibility w coordinate during transformation
	config->force_zero_w_term = false;

	// Use fixed sources (not from file)
	config->synthetic_sources = false;

	// Use fixed visibilities (not from file)
	config->synthetic_visibilities = false;

	// if using synthetic visibility creation, set this flag to
	// Gaussian distribute random visibility positions
	config->gaussian_distribution_sources = false;

	// Origin of Sources
	config->source_file = "../../data/500_synthetic_sources.csv";

	// Source of Visibilities
	config->vis_src_file    = "../../data/32_million_vis.csv";

	// Destination for processed visibilities
	config->vis_dest_file 	= "../../data/32_million_vis_output_test.csv";

	// Dimension of Fourier domain grid
	config->grid_size = 18000.0;

	// Fourier domain grid cell size in radians
	config->cell_size = 0.00000639708380288949;

	// Frequency of visibility uvw terms
	config->frequency_hz = 100e6;

	// Scalar for visibility coordinates
	config->uv_scale = config->grid_size * config->cell_size;

	// Range for synthetic visibility u coordinates
	config->min_u = -(config->grid_size / 2.0);
	config->max_u = config->grid_size / 2.0;

	// Range for synthetic visibility v coordinates
	config->min_v = -(config->grid_size / 2.0);
	config->max_v = config->grid_size / 2.0;

	// Range for synthetic visibility w coordinates
	config->min_w = config->min_v / 10;
	config->max_w = config->max_v / 10;

	// Number of CUDA blocks (gpu specific)
	config->gpu_num_blocks = 32;

	// Number of CUDA threads per block (updated when reading vis from file)
	config->gpu_num_threads = config->num_visibilities / config->gpu_num_blocks;

	// Enables/disables the printing of information during DFT
	config->enable_messages = true;
}

void extract_visibilities(Config *config, Source *sources, Visibility *visibilities, 
	Complex *vis_intensity, int num_visibilities)
{
	//Allocating GPU memory for visibility intensity
	PRECISION3 *device_sources;
	PRECISION3 *device_visibilities;
	PRECISION2 *device_intensities;

	if(config->enable_messages)
		printf(">>> UPDATE: Allocating GPU memory...\n\n");

	//copy the sources to the GPU.
	CUDA_CHECK_RETURN(cudaMalloc(&device_sources,  sizeof(PRECISION3) * config->num_sources));
	CUDA_CHECK_RETURN(cudaMemcpy(device_sources, sources, 
		config->num_sources * sizeof(PRECISION3), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	//copy the visibilities to the GPU
	CUDA_CHECK_RETURN(cudaMalloc(&device_visibilities,  sizeof(PRECISION3) * num_visibilities));
	CUDA_CHECK_RETURN(cudaMemcpy(device_visibilities, visibilities, 
		num_visibilities * sizeof(PRECISION3), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocate memory on GPU for storing extracted visibility intensities
	CUDA_CHECK_RETURN(cudaMalloc(&device_intensities,  sizeof(PRECISION2) * num_visibilities));
	cudaDeviceSynchronize();

	// Define number of blocks and threads per block on GPU
	dim3 kernel_blocks(config->gpu_num_blocks, 1, 1);
	dim3 kernel_threads(config->gpu_num_threads, 1, 1);

	if(config->enable_messages)
		printf(">>> UPDATE: Calling DFT GPU Kernel to create %d visibilities...\n\n", num_visibilities);

	//record events for timing kernel execution
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	direct_fourier_transform<<<kernel_threads, kernel_blocks>>>(device_visibilities,
		device_intensities, num_visibilities, device_sources, config->num_sources);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	if(config->enable_messages)
		printf(">>> UPDATE: DFT GPU Kernel Completed, Time taken %f mS...\n\n",milliseconds);

	CUDA_CHECK_RETURN(cudaMemcpy(vis_intensity, device_intensities, 
		num_visibilities * sizeof(PRECISION2), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	if(config->enable_messages)
		printf(">>> UPDATE: Copied Visibility Data back to Host - Completed...\n\n");

	// Clean up
	CUDA_CHECK_RETURN(cudaFree(device_intensities));
	CUDA_CHECK_RETURN(cudaFree(device_sources));
	CUDA_CHECK_RETURN(cudaFree(device_visibilities));
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

__global__ void direct_fourier_transform(const __restrict__ PRECISION3 *visibility, PRECISION2 *vis_intensity,
	const int vis_count, const PRECISION3 *sources, const int source_count)
{
	const int vis_indx = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_indx >= vis_count)
		return;

	PRECISION2 source_sum = MAKE_PRECISION2(0.0, 0.0);
	PRECISION term = 0.0;
	PRECISION w_correction = 0.0;
	PRECISION image_correction = 0.0;
	PRECISION theta = 0.0;
	PRECISION src_correction = 0.0;

	const PRECISION3 vis = visibility[vis_indx];
	PRECISION3 src;
	PRECISION2 theta_complex = MAKE_PRECISION2(0.0, 0.0);

	const double two_PI = CUDART_PI + CUDART_PI;
	// For all sources
	for(int src_indx = 0; src_indx < source_count; ++src_indx)
	{	
		src = sources[src_indx];
		
		// formula sqrt
		// term = sqrt(1.0 - (src.x * src.x) - (src.y * src.y));
		// image_correction = term;
		// w_correction = term - 1.0; 

		//approxiamation formula (unit test fails as less accurate)
		term = 0.5 * ((src.x * src.x) + (src.y * src.y));
		w_correction = -term;
		image_correction = 1.0 - term;

		src_correction = src.z / image_correction;

		theta = (vis.x * src.x + vis.y * src.y + vis.z * w_correction) * two_PI;
		sincos(theta, &(theta_complex.y), &(theta_complex.x));
		source_sum.x += theta_complex.x * src_correction;
		source_sum.y += -theta_complex.y * src_correction;
	}

	vis_intensity[vis_indx] = MAKE_PRECISION2(source_sum.x, source_sum.y);
}

void load_visibilities(Config *config, Visibility **visibilities, Complex **vis_intensity)
{
	if(config->synthetic_visibilities)
	{
		if(config->enable_messages)
			printf(">>> UPDATE: Using synthetic Visibilities...\n\n");

		*visibilities =  (Visibility*) calloc(config->num_visibilities, sizeof(Visibility));
		if(*visibilities == NULL)  return;

		*vis_intensity =  (Complex*) calloc(config->num_visibilities, sizeof(Complex));
		if(*vis_intensity == NULL)
		{	
			if(*visibilities) free(*visibilities);
			return;
		}

		PRECISION gaussian_u = 1.0;
		PRECISION gaussian_v = 1.0;
		PRECISION gaussian_w = 1.0;

		//try randomize visibilities in the center of the grid
		for(int vis_indx = 0; vis_indx < config->num_visibilities; ++vis_indx)
		{	
			if(config->gaussian_distribution_sources)
			{	
				gaussian_u = generate_sample_normal();
				gaussian_v = generate_sample_normal();
				gaussian_w = generate_sample_normal();
			}

			PRECISION u = random_in_range(config->min_u, config->max_u) * gaussian_u;
			PRECISION v = random_in_range(config->min_v, config->max_v) * gaussian_v;
			PRECISION w = random_in_range(config->min_w, config->max_w) * gaussian_w;
			
			(*visibilities)[vis_indx] = (Visibility) {
				.u = u / config->uv_scale,
				.v = v / config->uv_scale,
				.w = (config->force_zero_w_term) ? 0.0 : w / config->uv_scale
			};
		}
	}
	else // Reading visibilities from file
	{
		if(config->enable_messages)
			printf(">>> UPDATE: Using Visibilities from file...\n\n");

		FILE *file = fopen(config->vis_src_file, "r");
		if(file == NULL)
		{
			printf(">>> ERROR: Unable to locate visibilities file...\n\n");
			return;
		}

		// Reading in the counter for number of visibilities
		fscanf(file, "%d\n", &(config->num_visibilities));
		// Update gpu threads based on new number of visibilities (non-default)
		config->gpu_num_threads = config->num_visibilities / config->gpu_num_blocks;

		*visibilities = (Visibility*) calloc(config->num_visibilities, sizeof(Visibility));
		*vis_intensity =  (Complex*) calloc(config->num_visibilities, sizeof(Complex));

		// File found, but was memory allocated?
		if(*visibilities == NULL || *vis_intensity == NULL)
		{
			printf(">>> ERROR: Unable to allocate memory for visibilities...\n\n");
			if(file) fclose(file);
			if(*visibilities) free(*visibilities);
			if(*vis_intensity) free(*vis_intensity);
			return;
		}

		double u = 0.0;
		double v = 0.0;
		double w = 0.0;
		Complex brightness;
		double intensity = 0.0;

		// Used to scale visibility coordinates from wavelengths
		// to meters
		double wavelength_to_meters = config->frequency_hz / C;

		// Read in n number of visibilities
		for(int vis_indx = 0; vis_indx < config->num_visibilities; ++vis_indx)
		{
			// Read in provided visibility attributes
			// u, v, w, brightness (real), brightness (imag), intensity
			fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &u, &v, &w, 
				&(brightness.real), &(brightness.imaginary), &intensity);

			(*visibilities)[vis_indx] = (Visibility) {
				.u = u * wavelength_to_meters,
				.v = v * wavelength_to_meters,
				.w = (config->force_zero_w_term) ? 0.0 : w * wavelength_to_meters
			};
		}

		// Clean up
		fclose(file);
		if(config->enable_messages)
			printf(">>> UPDATE: Successfully loaded %d visibilities from file...\n\n",config->num_visibilities);
	}
}

void load_sources(Config *config, Source **sources)
{
	if(config->synthetic_sources)
	{
		if(config->enable_messages)
			printf(">>> UPDATE: Using synthetic Sources...\n\n");

		*sources = (Source*) calloc(config->num_sources, sizeof(Source));
		if(*sources == NULL) return;

		for(int src_indx = 0; src_indx < config->num_sources; ++src_indx)
		{
			(*sources)[src_indx] = (Source) {
				.l = random_in_range(config->min_u, config->max_u) * config->cell_size,
				.m = random_in_range(config->min_v, config->max_v) * config->cell_size,
				.intensity = 1.0
			};
		}

		if(config->enable_messages)
			printf(">>> UPDATE: Successfully loaded %d synthetic sources..\n\n",config->num_sources);
	}
	else // Reading Sources from file
	{
		if(config->enable_messages)
			printf(">>> UPDATE: Using Sources from file...\n\n");

		FILE *file = fopen(config->source_file, "r");
		// Unable to open file
		if(file == NULL)
		{	
			printf(">>> ERROR: Unable to load sources from file...\n\n");
			return;
		}

		fscanf(file, "%d\n", &(config->num_sources));
		*sources = (Source*) calloc(config->num_sources, sizeof(Source));
		if(*sources == NULL)
	 	{
	 		fclose(file);
	 		return;
		}

		PRECISION temp_l = 0.0;
		PRECISION temp_m = 0.0;
		PRECISION temp_intensity = 0.0;

		for(int src_indx = 0; src_indx < config->num_sources; ++src_indx)
		{
			fscanf(file, "%lf %lf %lf\n", &temp_l, &temp_m, &temp_intensity);

			(*sources)[src_indx] = (Source) {
				.l = temp_l * config->cell_size,
				.m = temp_m * config->cell_size,
				.intensity = temp_intensity
			};
		}

		// Clean up
		fclose(file);
		if(config->enable_messages)
			printf(">>> UPDATE: Successfully loaded %d sources from file..\n\n",config->num_sources);
	}
}


void save_visibilities(Config *config, Visibility *visibilities, Complex *vis_intensity)
{
	// Save visibilities to file
	FILE *file = fopen(config->vis_dest_file, "w");
	// Unable to open file
	if(file == NULL)
	{
		printf(">>> ERROR: Unable to save visibilities to file...\n\n");
		return;
	}

	if(config->enable_messages)
		printf(">>> UPDATE: Writing visibilities to file...\n\n");

	// Record number of visibilities
	fprintf(file, "%d\n", config->num_visibilities);
	
	// Used to scale visibility coordinates from meters to
	// wavelengths (useful for gridding, inverse DFT etc.)
	double meters_to_wavelengths = config->frequency_hz / C;

	// Record individual visibilities
	for(int vis_indx = 0; vis_indx < config->num_visibilities; ++vis_indx)
	{
		// u, v, w, real, imag, weight (intensity)
		fprintf(file, "%f %f %f %f %f %f\n", 
			visibilities[vis_indx].u / meters_to_wavelengths,
			visibilities[vis_indx].v / meters_to_wavelengths,
			visibilities[vis_indx].w / meters_to_wavelengths,
			vis_intensity[vis_indx].real,
			vis_intensity[vis_indx].imaginary,
			1.0); // static intensity (for now)
	}

	// Clean up
	fclose(file);
	if(config->enable_messages)
		printf(">>> UPDATE: Completed writing of visibilities to file...\n\n");
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}

PRECISION random_in_range(PRECISION min, PRECISION max)
{
	PRECISION range = (max - min);
	PRECISION div = RAND_MAX / range;
	return min + (rand() / div);
}

PRECISION generate_sample_normal()
{
	PRECISION u = ((PRECISION) rand() / RAND_MAX) * 2.0 - 1.0;
	PRECISION v = ((PRECISION) rand() / RAND_MAX) * 2.0 - 1.0;
	PRECISION r = u * u + v * v;
	if(r <= 0.0 || r > 1.0)
		return generate_sample_normal();
	return u * sqrt(-2.0 * log(r) / r);
}

//**************************************//
//      UNIT TESTING FUNCTIONALITY      //
//**************************************//

void unit_test_init_config(Config *config)
{
	config->num_sources 					= 1;
	config->num_visibilities 				= 1;
	config->source_file 					= "../unit_test_20_synth_sources.csv";
	config->vis_src_file    				= "../unit_test_1k_vis_input.csv";
	config->vis_dest_file 					= "../unit_test_1k_vis_output.csv";
	config->synthetic_sources 				= false;
	config->synthetic_visibilities 			= false;
	config->gaussian_distribution_sources 	= false;
	config->force_zero_w_term 				= false;
	config->grid_size 						= 18000;
	config->cell_size 						= 0.00000639708380288949;
	config->frequency_hz 					= 100e6;
	config->uv_scale 						= config->grid_size * config->cell_size;
	config->min_u 							= -(config->grid_size / 2.0);
	config->max_u 							= config->grid_size / 2.0;
	config->min_v 							= -(config->grid_size / 2.0);
	config->max_v 							= config->grid_size / 2.0;
	config->min_w 							= config->min_v / 10;
	config->max_w 							= config->max_v / 10;
	config->gpu_num_blocks					= 1;
	config->gpu_num_threads					= 1;
	config->enable_messages 				= false;
}

double unit_test_generate_approximate_visibilities(void)
{
	// used to invalidate the unit test
	double error = DBL_MAX;

	Config config;
	unit_test_init_config(&config);

	// Read in test sources
	Source *sources = NULL;
	load_sources(&config, &sources);
	if(sources == NULL)
		return error;

	// Read in test visibilities and process
	FILE *file = fopen(config.vis_src_file, "r");
	if(file == NULL)
	{
		if(sources) free(sources);
		return error;
	}

	fscanf(file, "%d\n", &(config.num_visibilities));

	double u = 0.0;
	double v = 0.0;
	double w = 0.0;
	double intensity = 0.0;
	double difference = 0.0;
	double wavelength_to_meters = config.frequency_hz / C;
	Complex brightness = (Complex) {.real = 0.0, .imaginary = 0.0};
	Complex test_vis_intensity;
	Visibility approx_visibility[1]; // testing one at a time
	Complex approx_vis_intensity[1]; // testing one at a time

	for(int vis_indx = 0; vis_indx < config.num_visibilities; ++vis_indx)
	{
		fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &u, &v, &w, 
			&(brightness.real), &(brightness.imaginary), &intensity);

		test_vis_intensity.real      = brightness.real;
		test_vis_intensity.imaginary = brightness.imaginary;

		approx_visibility[0] = (Visibility) {
			.u = u * wavelength_to_meters,
			.v = v * wavelength_to_meters,
			.w = w * wavelength_to_meters
		};

		approx_vis_intensity[0] = (Complex) {
			.real      = 0.0,
			.imaginary = 0.0
		};

		// Measure one visibility brightness from n sources
		extract_visibilities(&config, sources, approx_visibility, approx_vis_intensity, 1);

		double current_difference = sqrt(pow(approx_vis_intensity[0].real
			-test_vis_intensity.real, 2.0)
			+ pow(approx_vis_intensity[0].imaginary
			-test_vis_intensity.imaginary, 2.0));

		if(current_difference > difference)
			difference = current_difference;
	}

	// Clean up
	fclose(file);
	if(sources) free(sources);

	printf(">>> INFO: Measured maximum difference of evaluated visibilities is %f\n", difference);

	return difference;
}