
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

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <sys/time.h>

#include "direct_fourier_transform.h"

int main(int argc, char **argv)
{
	printf("============================================================================\n");
	printf(">>> AUT HPC Research Laboratory - Direct Fourier Transform (GPU version) <<<\n");
	printf("============================================================================\n\n");

	// Seed random from time
	srand(time(NULL));

	Config config;
	init_config(&config);

	Source *sources = NULL;
	load_sources(&config, &sources);
	if(sources == NULL)
	{	
		printf(">>> ERROR: Source memory was unable to be allocated...\n\n");
		return EXIT_FAILURE;
	}

	Visibility *visibilities = NULL;
	Complex *vis_intensity = NULL;
	load_visibilities(&config, &visibilities, &vis_intensity);

	if(visibilities == NULL || vis_intensity == NULL)
	{	
		printf(">>> ERROR: Visibility memory was unable to be allocated...\n\n");
		if(sources)      	   free(sources);
		if(visibilities)       free(visibilities);
		if(vis_intensity)      free(vis_intensity);
		return EXIT_FAILURE;
	}

	extract_visibilities(&config, sources, visibilities, vis_intensity, config.num_visibilities);

	// Save visibilities to file
	//save_visibilities(&config, visibilities, vis_intensity);

	// Clean up
	if(visibilities)  free(visibilities);
	if(sources)       free(sources);
	if(vis_intensity) free(vis_intensity);

	printf(">>> INFO: Direct Fourier Transform operations complete, exiting...\n\n");

	return EXIT_SUCCESS;
}
