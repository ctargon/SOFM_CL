#include <time.h>
#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/opencl.h>
#endif

#include "../../common/inc/AOCL_Utils.h"

#include "sofm.h"

using namespace aocl_utils;



// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
cl_device_id *device; 
cl_context context = NULL;
cl_command_queue queue; 
cl_program program = NULL;
cl_kernel bmu_kernel;
cl_kernel update_kernel; 

cl_mem weights_buf;
cl_mem dataset_buf;
cl_mem distances_buf;


// global problem parameters
int dim_x;
int dim_y;
int n_features;

int iters;
float init_lr;
float lr;

float max_dim;
float init_radius;
float radius;
float time_delay;

// global data pointers and params
float *data;
float *weights;
int num_data_examples;

char *type_data = NULL;


// openCL kernel
char *kernelBuf = NULL;


// Function prototypes
void read_kernel_file(char *file);
bool init_opencl();
void init_problem();
void run();
void cleanup();


#define NUM_SHORTS 10000000


// Entry point.
int main(int argc, char **argv) {
	time_t t;

	srand((unsigned) time(&t));

	if (argc != 5)
	{
		printf("Expecting: ./sofm_cl ./path/to/kernel gtex/color dim_x dim_y\n");
		return -1;
	}

	type_data = argv[2];

	dim_x = atoi(argv[3]);
	dim_y = atoi(argv[4]);

	// read kernel file
	read_kernel_file(argv[1]);


	// Initialize the problem data.
	init_problem();

	// Initialize OpenCL.
	if(!init_opencl()) 
	{
		printf("Initializing OpenCL failed.\n");
		return -1;
	}

	//printf("yeet dawg\n");


	// Run the kernel.
	run();

	// Free the resources allocated
	cleanup();

	return 0;
}



/////// HELPER FUNCTIONS ///////

void read_kernel_file(char *file)
{
	FILE *fptr = fopen(file, "r");
	fseek(fptr, 0, SEEK_END);
	int kernelSize = ftell(fptr);
	rewind(fptr);

	kernelBuf = (char *) malloc (sizeof(char) * kernelSize + 1);
	kernelBuf[kernelSize] = '\0';
	fread(kernelBuf, sizeof(char), kernelSize, fptr);
	fclose(fptr);
}



//Initialize data for the problem.
void init_problem() {

	//int i;
	if (strcmp(type_data, "gtex") == 0)
	{
		n_features = 36;
		iters = 10000;
		init_lr = lr = 0.1;

		max_dim = max(dim_x, dim_y);
		init_radius = radius = (float) max_dim / 2.0;

		time_delay = (float) iters / log(init_radius);

		num_data_examples = 8157;
		data = load_data_file("./data/myc_targets_v2_train_data.dat", n_features, num_data_examples);
	}

	if (strcmp(type_data, "color") == 0)
	{
		n_features = 3;
		iters = 10000;
		init_lr = lr = 0.1;

		max_dim = max(dim_x, dim_y);
		init_radius = radius = (float) max_dim / 2.0;

		time_delay = (float) iters / log(init_radius);

		num_data_examples = 1000;
		data = load_rand_colors(num_data_examples, n_features);
	}



	weights = initialize_weights(dim_x, dim_y, n_features);
 
}




// Initializes the OpenCL objects.
bool init_opencl() {
	cl_int status;
	cl_int err;

	printf("Initializing OpenCL\n");

	// if(!setCwdToExeDir()) {
   //      printf("uh oh");
   //  		return false;
	// }

	// Get the OpenCL platform.
	platform = findPlatform("NVIDIA CUDA");
	if(platform == NULL) 
	{
		printf("ERROR: Unable to find Altera OpenCL platform.\n");
		return false;
	}

	// Query the available OpenCL device.
	device = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
	//printf("Platform: %s\n", getPlatformName(platform).c_str());
	//printf("Using %d device(s)\n", num_devices);
	//printf("  %s\n", getDeviceName(*device).c_str());

	// Create the context.
	context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the program for all device. Use the first device as the
	// representative device (assuming all device are of the same type).
	// std::string binary_file = getBoardBinaryFile("ripple_adder_cl", *device);
	// //printf("Using AOCX: %s\n", binary_file.c_str());
	// program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);
	program = clCreateProgramWithSource(context, 1, 
		(const char **) & kernelBuf, NULL, &err);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);

	if (status == CL_BUILD_PROGRAM_FAILURE) 
	{
	    // Determine the size of the log
	    size_t log_size;
	    clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	    // Allocate memory for the log
	    char *log = (char *) malloc(log_size);

	    // Get the log
	    clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

	    // Print the log
	    printf("%s\n", log);
	}

	checkError(status, "Failed to build program");

	// Command queue.
	queue = clCreateCommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Kernel.
	const char *kernel_name = "find_bmu";
	bmu_kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create bmu_kernel");

	const char *kernel_name2 = "update_weights";
	update_kernel = clCreateKernel(program, kernel_name2, &status);
	checkError(status, "Failed to create bmu_kernel");

	// //Input buffer.
	// input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
	// 		sizeof(unsigned short) * N, NULL, &status);
	// checkError(status, "Failed to create buffer for input");

	// input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
	// 		sizeof(unsigned short) * N, NULL, &status);
	// checkError(status, "Failed to create buffer for input");

	// // Output buffer.
	// output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
	// 			sizeof(unsigned short) * N, NULL, &status);
	// checkError(status, "Failed to create buffer for output");

	weights_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(float) * dim_x * dim_y * n_features,
			NULL, &status);
	checkError(status, "Failed to create buffer for weights");

	dataset_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
			sizeof(float) * n_features * num_data_examples,
			NULL, &status);
	checkError(status, "Failed to create buffer for dataset");

	distances_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(float) * dim_x * dim_y, NULL, &status);

	return true;
}



void run() 
{
	int i, j, rand_int, bmu_idx = -1;
	float *distances = (float *) malloc (sizeof(float) * dim_x * dim_y);
	cl_int status;

	// Launch the problem for each device.
	cl_event kernel_event;
	cl_event finish_event;
	cl_event write_event;

	save_weights((char *)"./weights/init_opencl_sofm_weights.dat", weights, dim_x, dim_y, n_features);

	status = clEnqueueWriteBuffer(queue, weights_buf, CL_FALSE,
				0, sizeof(float) * dim_x * dim_y * n_features, 
				weights, 0, NULL, &write_event);
	checkError(status, "Failed to write input buffer");

	status = clEnqueueWriteBuffer(queue, dataset_buf, CL_FALSE,
				0, sizeof(float) * num_data_examples * n_features, 
				data, 0, NULL, &write_event);
	checkError(status, "Failed to write input buffer");
	
	status = clEnqueueWriteBuffer(queue, distances_buf, CL_FALSE,
				0, sizeof(float) * dim_x * dim_y, 
				distances, 0, NULL, &write_event);
	checkError(status, "Failed to write input buffer"); 


	// Enqueue kernel.
	// Use a global work size corresponding to the number of elements to add
	// for this device.
	// 
	// We don't specify a local work size and let the runtime choose
	// (it'll choose to use one work-group with the same size as the global
	// work-size).
	//
	// Events are used to ensure that the kernel is not launched until
	// the writes to the input buffers have completed.
	const size_t localSize = 256;
	//const size_t global_work_size = N;
	const size_t global_work_size = ceil((dim_x * dim_y)/(float)localSize)*localSize;
	//printf("Launching for device %d (%d elements)\n", 0, global_work_size);

	rand_int = rand() % num_data_examples;

	// Set kernel arguments for the find_bmu kernel
	unsigned argi = 0;
	status = clSetKernelArg(bmu_kernel, argi++, sizeof(cl_mem), &dataset_buf);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(bmu_kernel, argi++, sizeof(cl_mem), &weights_buf);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(bmu_kernel, argi++, sizeof(cl_mem), &distances_buf);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(bmu_kernel, argi++, sizeof(int), &dim_x);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(bmu_kernel, argi++, sizeof(int), &dim_y);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(bmu_kernel, argi++, sizeof(int), &rand_int);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(bmu_kernel, argi++, sizeof(int), &n_features);
	checkError(status, "Failed to set argument %d", argi - 1);

	// set arguments for the update kernel!
	argi = 0;
	status = clSetKernelArg(update_kernel, argi++, sizeof(cl_mem), &dataset_buf);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(cl_mem), &weights_buf);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(cl_mem), &distances_buf);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(int), &bmu_idx);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(int), &dim_x);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(int), &dim_y);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(int), &rand_int);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(int), &n_features);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(float), &radius);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(update_kernel, argi++, sizeof(float), &lr);
	checkError(status, "Failed to set argument %d", argi - 1);


	double start_time = getCurrentTimestamp();

	for (i = 0; i < iters; i++)
	{
		if (i % 1000 == 0)
			printf("iter %d\n", i);

		rand_int = rand() % num_data_examples;

		// get new radius and new learning rate
		radius = decay_radius(init_radius, i, time_delay);
		lr = decay_lr(init_lr, i, iters);

		// set new kernel argument for random number (decides input vector)
		status = clSetKernelArg(bmu_kernel, 5, sizeof(int), &rand_int);
		checkError(status, "Failed to set argument %d", 5);


		// launch kernel with set arguments
		status = clEnqueueNDRangeKernel(queue, bmu_kernel, 1, NULL,
			  &global_work_size, &localSize, 1, &write_event, &kernel_event);
		checkError(status, "Failed to launch kernel");

		clWaitForEvents(num_devices, &kernel_event);

		// Read the result. This the final operation.
		status = clEnqueueReadBuffer(queue, distances_buf, CL_FALSE,
			  0, sizeof(float) * dim_x * dim_y, distances, 1, &kernel_event, &finish_event);
		// status = clEnqueueReadBuffer(queue, bmu_idx_buf, CL_FALSE,
		// 	  0, sizeof(int), &bmu_idx, 1, &kernel_event, &finish_event);

		// Wait for all devices to finish.
		clWaitForEvents(num_devices, &finish_event);	
		//clFinish(queue);

		float min = 999999999.9;
		bmu_idx = -1;
		for (j = 0; j < dim_x * dim_y; j++)
		{
			if (distances[j] < min)
			{
				min = distances[j];
				bmu_idx = j;
			}
		}

		// status = clEnqueueWriteBuffer(queue, bmu_idx_buf, CL_FALSE,
		// 			0, sizeof(int), 
		// 			&bmu_idx, 0, NULL, &write_event);
		// checkError(status, "Failed to write input buffer"); 

		// // Wait for all devices to finish.
		// clWaitForEvents(num_devices, &write_event);	

		// set new kernel argument for random number (decides input vector)
		status = clSetKernelArg(update_kernel, 3, sizeof(int), &bmu_idx);
		checkError(status, "Failed to set argument %d", 3);
		status = clSetKernelArg(update_kernel, 6, sizeof(int), &rand_int);
		checkError(status, "Failed to set argument %d", 6);
		status = clSetKernelArg(update_kernel, 8, sizeof(float), &radius);
		checkError(status, "Failed to set argument %d", 8);
		status = clSetKernelArg(update_kernel, 9, sizeof(float), &lr);
		checkError(status, "Failed to set argument %d", 9);

		// launch kernel with set arguments
		status = clEnqueueNDRangeKernel(queue, update_kernel, 1, NULL,
			  &global_work_size, &localSize, 1, &write_event, &kernel_event);
		checkError(status, "Failed to launch kernel");

		clWaitForEvents(num_devices, &kernel_event);


		//printf("idx is %d %d\n", bmu_idx / dim_x, bmu_idx % dim_y);	
	}



	clWaitForEvents(num_devices, &finish_event);
	status = clEnqueueReadBuffer(queue, weights_buf, CL_FALSE,
			  0, sizeof(float) * dim_x * dim_y * n_features, weights, 1, &kernel_event, &finish_event);


	save_weights((char *)"./weights/final_opencl_sofm_weights.dat", weights, dim_x, dim_y, n_features);



	double end_time = getCurrentTimestamp();

	// Wall-clock time taken.
	printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

	// Get kernel times using the OpenCL event profiling API.
	cl_ulong time_ns = getStartEndTime(kernel_event);
	printf("Kernel time (device 0): %0.3f ms\n", double(time_ns) * 1e-6);

 
	// Measure host execution time.
	// start_time = getCurrentTimestamp();

	// for(i = 0; i < N; i++)
	// {
	// 	c_output[i] = inputs_a[i] + inputs_b[i];
	// }

	// end_time = getCurrentTimestamp();



	printf("\nC time: %0.3f ms\n", (end_time - start_time) * 1e3);

	// Release all events.
	clReleaseEvent(kernel_event);
	clReleaseEvent(finish_event);

}





// Free the resources allocated during initialization
void cleanup() 
{
	if(bmu_kernel) {
		clReleaseKernel(bmu_kernel);
	}
	if(update_kernel) {
		clReleaseKernel(update_kernel);
	}
	if(queue) {
		  clReleaseCommandQueue(queue);
	}

	if(weights_buf) {
		  clReleaseMemObject(weights_buf);
	}

	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}
}














