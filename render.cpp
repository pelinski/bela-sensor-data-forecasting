/*
bela-sensor-data-forecasting/render.cpp, based on C++ Real-Time Audio Programming with Bela - Lecture 18: Phase vocoder, part 1
fft-overlap-add-threads: overlap-add framework doing FFT in a low-priority thread
*/

#include <Bela.h>
#include <libraries/Fft/Fft.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "AppOptions.h"


// IO-related variables
const int gInputWindowSize = 32;	// Input window size in samples
const int gOutputWindowSize = 3*32;
const int gHopSize = 3*32;	// How often we calculate a window

// Circular buffer and pointer for assembling a window of samples
const int gBufferSize = 128;
std::vector<float> gInputBuffer(gBufferSize);
int gInputBufferPointer = 0;
int gHopCounter = 0;

// Circular buffer for collecting the output of the overlap-add process
std::vector<float> gOutputBuffer(gBufferSize);
int gOutputBufferWritePointer = 2*gHopSize;	// Need one extra hop of latency to run in second thread
int gOutputBufferReadPointer = 0;

// Thread for inference
AuxiliaryTask gInferenceTask;
int gCachedInputBufferPointer = 0;

// Tensorflow-related variables
std::unique_ptr<tflite::FlatBufferModel> model;
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;

// Sensor-related variables
unsigned int gAudioFramesPerAnalogFrame;
int gSensorCh = 0;

// Audio-related variables
float gPhase_inSin = 0.0;
float gPhase_outSin = 0.0;
float gInverseSampleRate;
float gFrequency_inSin = 440.0;
float gFrequency_outSin = 440.0;


void inference_task_background(void *);

// setup() runs at he beggining of the project execution, before any audio processing starts. Resources should be allocated here.
bool setup(BelaContext *context, void *userData)
{

	printf("analog sample rate: %.1f\n", context->analogSampleRate);

	// Better to calculate the inverse sample rate here and store it in a variable so it can be reused
	gInverseSampleRate = 1.0 / context->audioSampleRate;
	
	// Set up the thread for the inference
	gInferenceTask = Bela_createAuxiliaryTask(inference_task_background, 50, "bela-inference");

	// rate of audio frames per analog frame
	if (context->analogFrames) gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;

	// Load tflite model (passed through -m)
    AppOptions *opts = reinterpret_cast<AppOptions *>(userData);

    model = tflite::FlatBufferModel::BuildFromFile(opts->modelPath.c_str());
        if(!model){
        printf("Failed to mmap model\n");
        return false;
    }

	// Build Tf interpreter
	tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

	// Allocate IO tensors
	interpreter->AllocateTensors();

	return true;
}

// inference_task() handles the inference once the buffer has been assembled
void inference_task(std::vector<float> const& inBuffer, unsigned int inPointer, std::vector<float>& outBuffer, unsigned int outPointer)
{
	static std::vector<float> unwrappedBuffer(gOutputWindowSize);	// Container to hold the unwrapped values
	
	// Copy buffer into FFT input, starting one window ago
	for(int n = 0; n < gInputWindowSize; n++) {
		// Use modulo arithmetic to calculate the circular buffer index
		int circularBufferIndex = (inPointer + n - gInputWindowSize + gBufferSize) % gBufferSize;
		unwrappedBuffer[n] = inBuffer[circularBufferIndex];
	}


	float* input = interpreter->typed_inSinput_tensor<float>(0);
	for (int i=0; i<unwrappedBuffer.size(); i++){
		*input = unwrappedBuffer[i];
		input++;
	}

	interpreter->Invoke();

    float* output = interpreter->typed_outSinput_tensor<float>(0);
		
	// Add timeDomainOut into the output buffer starting at the write pointer
	for(int n = 0; n < gOutputWindowSize; n++) {
		int circularBufferIndex = (outPointer + n) % gBufferSize;
		float tmp = *output;
		outBuffer[circularBufferIndex] = tmp;
		output++;
	}
}

// This function runs in an auxiliary task on Bela, calling inference_task
void inference_task_background(void *)
{
	inference_task(gInputBuffer, gCachedInputBufferPointer, gOutputBuffer, gOutputBufferWritePointer);

	// Update the output buffer write pointer to start at the next hop
	gOutputBufferWritePointer = (gOutputBufferWritePointer + gHopSize) % gBufferSize;
}

// render() is called for each block of samples
void render(BelaContext *context, void *userData)
{

	for(unsigned int n = 0; n < context->audioFrames; n++) {

        // Read the sensor value
        float in = analogRead(context, n/gAudioFramesPerAnalogFrame, gSensorCh);

		// Sensor value is amplitude of inSin
		float inSin = in * sinf(gPhase_inSin);
		// Update and wrap phase 
		gPhase_inSin += 2.0f * (float)M_PI * gFrequency_inSin * gInverseSampleRate;
		if(gPhase_inSin > M_PI)
			gPhase_inSin -= 2.0f * (float)M_PI;

		// Store the sample ("in") in a buffer (input window of the model)
		// Increment the pointer and when full window has been 
		// assembled, call inference_task()
		gInputBuffer[gInputBufferPointer++] = in;
		if(gInputBufferPointer >= gBufferSize) {
			// Wrap the circular buffer
			// Notice: this is not the condition for starting a new inference
			gInputBufferPointer = 0;
		}

		// Get the output sample from the output buffer
		float out = gOutputBuffer[gOutputBufferReadPointer];
		// out is the amplitude of outSin
		float outSin = out * sinf(gPhase_outSin);
		gPhase_outSin += 2.0f * (float)M_PI * gFrequency_outSin * gInverseSampleRate;
		if(gPhase_outSin > M_PI)
			gPhase_outSin -= 2.0f * (float)M_PI;
		
		// Increment the read pointer in the output cicular buffer
		if(++gOutputBufferReadPointer >= gBufferSize)
			gOutputBufferReadPointer = 0;
		
		// Increment the hop counter and start a new inference if we've reached the hop size
		if(++gHopCounter >= gHopSize) {
			gHopCounter = 0;
			gCachedInputBufferPointer = gInputBufferPointer;
			Bela_scheduleAuxiliaryTask(gInferenceTask);
		}

		// Write the audio input to left channel, output to the right channel
		audioWrite(context, n, 0, inSin);
		audioWrite(context, n, 1, outSin);
	}
}

// cleanup() runs at the end of the program before it exits
void cleanup(BelaContext *context, void *userData)
{

}
