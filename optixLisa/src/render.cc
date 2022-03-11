#include "render.hh"

#include <optix_stubs.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <sutil/GLDisplay.h>

void save_image(sutil::CUDAOutputBuffer<uchar4> &output_buffer, const RendererParams &params) {
  sutil::ImageBuffer buffer;
  buffer.data         = output_buffer.getHostPointer();
  buffer.width        = output_buffer.width();
  buffer.height       = output_buffer.height();
  buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

  sutil::saveImage(params.output_image, buffer, false);
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, RendererState& state )
{
  // Launch
  uchar4* result_buffer_data = output_buffer.map();
  state.params.frame_buffer  = result_buffer_data;
  CUDA_CHECK( cudaMemcpyAsync(
              reinterpret_cast<void*>( state.d_params ),
              &state.params, sizeof( OptixParams ),
              cudaMemcpyHostToDevice, state.stream
              ) );

  OPTIX_CHECK( optixLaunch(
              state.pipeline,
              state.stream,
              reinterpret_cast<CUdeviceptr>( state.d_params ),
              sizeof( OptixParams ),
              &state.sbt,
              state.params.width,   // launch width
              state.params.height,  // launch height
              1                     // launch depth
              ) );
  output_buffer.unmap();
  CUDA_SYNC_CHECK();
}

void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
  // Display
  int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
  int framebuf_res_y = 0;  //
  glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
  gl_display.display(
          output_buffer.width(),
          output_buffer.height(),
          framebuf_res_x,
          framebuf_res_y,
          output_buffer.getPBO()
          );
}

void display(RendererState &state, const RendererParams &params) {
  GLFWwindow* window = sutil::initUI( "optixLisa", state.params.width, state.params.height );
  glfwSetKeyCallback( window, keyCallback );
  glfwSetWindowUserPointer( window, &state.params );
  //
  // Render loop
  //
  sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
  sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type,
                                                state.params.width,
                                                state.params.height);

  output_buffer.setStream( state.stream );
  sutil::GLDisplay gl_display;

  std::chrono::duration<double> state_update_time( 0.0 );
  std::chrono::duration<double> render_time( 0.0 );
  std::chrono::duration<double> display_time( 0.0 );

  do
  {
    glfwPollEvents();


    auto t0 = std::chrono::steady_clock::now();
    launchSubframe( output_buffer, state );
    auto t1 = std::chrono::steady_clock::now();
    render_time += t1 - t0;
    t0 = t1;

    displaySubframe( output_buffer, gl_display, window );
    t1 = std::chrono::steady_clock::now();
    display_time += t1 - t0;

    sutil::displayStats( state_update_time, render_time, display_time );
    char msg[50];
    sprintf(msg, "nb sample   : %8d", state.params.subframe_index * state.params.samples_per_launch);
    sutil::beginFrameImGui();
    sutil::displayText(msg, 10, 75);
    sutil::endFrameImGui();

    glfwSwapBuffers( window );

    state.params.subframe_index++;

    if (state.params.subframe_index * state.params.samples_per_launch >= params.num_samples) {
      save_image(output_buffer, params);
      break;
    }

  } while( !glfwWindowShouldClose( window ) );
  CUDA_SYNC_CHECK();
}

void render(RendererState &state, const RendererParams &params) {
  sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::ZERO_COPY;
  sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type,
                                              state.params.width,
                                              state.params.height);
  while(state.params.subframe_index * state.params.samples_per_launch < params.num_samples) {
    launchSubframe(output_buffer, state);
    state.params.subframe_index++;
  }
  save_image(output_buffer, params);
}