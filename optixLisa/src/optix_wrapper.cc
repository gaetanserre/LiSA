#include "optix_wrapper.hh"
#include <string.h>
#include <iomanip>

#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sutil/sutil.h>
#include <sutil/Exception.h>
#include <sutil/vec_math.h>
#include <sutil/Camera.h>

template <typename T>
struct Record
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
  std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

namespace optix_wrapper {

  /**** INIT OPTIX/CONTEXT ****/
  
  void init_optix(RendererState &state) {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    state.context = context;
  }


  /**** MESH HANDLER ****/

  void create_mesh_handler(RendererState &state,
                            const float3* vertices,
                            const float3* normals,
                            const int* mat_indices,
                            const int num_materials,
                            const int num_vertices)
  {
    //
    // copy mesh data to device
    //
    const size_t vertices_size_in_bytes = num_vertices * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_vertices), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
              reinterpret_cast<void*>(state.d_vertices),
              vertices, vertices_size_in_bytes,
              cudaMemcpyHostToDevice
              ));

    CUdeviceptr  d_mat_indices             = 0;
    const size_t mat_indices_size_in_bytes = num_vertices * sizeof(int);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_mat_indices),
                mat_indices,
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
                ));
    
    const size_t normals_size_in_bytes = num_vertices * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_normals), normals_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
              reinterpret_cast<void*>(state.d_normals),
              normals, normals_size_in_bytes,
              cudaMemcpyHostToDevice
              ));

    //
    // Build triangle GAS
    //
    uint32_t triangle_input_flags[num_materials];
    // One per SBT record for this build input
    for (int i = 0; i < num_materials; i++)
      triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof(float3);
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>(num_vertices);
    triangle_input.triangleArray.vertexBuffers               = &state.d_vertices;
    triangle_input.triangleArray.flags                       = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords               = num_materials;
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(int);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(int);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &triangle_input,
                1,  // num_build_inputs
                &gas_buffer_sizes
                ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    // output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
              reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
              compactedSizeOffset + 8
              ));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
                state.context,
                0,                                  // CUDA stream
                &accel_options,
                &triangle_input,
                1,                                  // num build inputs
                d_temp_buffer,
                gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size,
                gas_buffer_sizes.outputSizeInBytes,
                &state.d_gas_handler,
                &emitProperty,                      // emitted property list
                1                                   // num emitted properties
                ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));
  }


  /**** MODULES ****/

  void create_module(RendererState &state) {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 2;
    state.pipeline_compile_options.numAttributeValues    = 2;

    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "shader.cu", inputSize);

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &state.ptx_module
                ));
  }


  /**** PROGRAMS ****/

  void create_programs(RendererState &state) {
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof( log );

    {
      OptixProgramGroupDesc raygen_prog_group_desc    = {};
      raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygen_prog_group_desc.raygen.module            = state.ptx_module;
      raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

      OPTIX_CHECK_LOG(optixProgramGroupCreate(
                  state.context, &raygen_prog_group_desc,
                  1,  // num program groups
                  &program_group_options,
                  log,
                  &sizeof_log,
                  &state.raygen_prog_group
                  ));
    }

    {
      OptixProgramGroupDesc miss_prog_group_desc  = {};
      miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
      miss_prog_group_desc.miss.module            = state.ptx_module;
      miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
      sizeof_log                                  = sizeof( log );
      OPTIX_CHECK_LOG(optixProgramGroupCreate(
                  state.context, &miss_prog_group_desc,
                  1,  // num program groups
                  &program_group_options,
                  log,
                  &sizeof_log,
                  &state.radiance_miss_group
                  ));

      memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
      miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
      miss_prog_group_desc.miss.module            = state.ptx_module;
      miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";
      sizeof_log                                  = sizeof( log );
      OPTIX_CHECK_LOG(optixProgramGroupCreate(
                  state.context, &miss_prog_group_desc,
                  1,  // num program groups
                  &program_group_options,
                  log,
                  &sizeof_log,
                  &state.occlusion_miss_group
                  ));
    }

    {
      OptixProgramGroupDesc hit_prog_group_desc        = {};
      hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
      hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
      sizeof_log                                       = sizeof( log );
      OPTIX_CHECK_LOG(optixProgramGroupCreate(
                  state.context,
                  &hit_prog_group_desc,
                  1,  // num program groups
                  &program_group_options,
                  log,
                  &sizeof_log,
                  &state.radiance_hit_group
                  ));

      memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
      hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
      hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
      sizeof_log                                       = sizeof( log );
      OPTIX_CHECK(optixProgramGroupCreate(
                  state.context,
                  &hit_prog_group_desc,
                  1,  // num program groups
                  &program_group_options,
                  log,
                  &sizeof_log,
                  &state.occlusion_hit_group
                  ));
    }
  }


  /**** PIPELINE ****/

  void create_pipeline(RendererState &state) {
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.radiance_miss_group,
        state.occlusion_miss_group,
        state.radiance_hit_group,
        state.occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = 2;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG( optixPipelineCreate(
                state.context,
                &state.pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &state.pipeline
                ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup prog : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(prog, &stack_sizes));
    }

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth,
                max_cc_depth,
                max_dc_depth,
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size
                ));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
                state.pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth
                ));

  }


  /**** SHADER BINDING TABLES ****/

  typedef Record<RayGenData>   RayGenRecord;
  typedef Record<MissData>     MissRecord;
  typedef Record<HitGroupData> HitGroupRecord;
  void create_shaders_binding_table(RendererState &state,
                                    const Material* materials,
                                    const int num_materials)
  {
    const int RAY_TYPE_COUNT = 2;
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc( reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ));


    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &d_miss_records ), miss_record_size * RAY_TYPE_COUNT));

    MissRecord ms_sbt[RAY_TYPE_COUNT];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group,  &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);
    OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_miss_group, &ms_sbt[1]));
    ms_sbt[1].data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_miss_records),
                ms_sbt,
                miss_record_size * RAY_TYPE_COUNT,
                cudaMemcpyHostToDevice
                ));

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>( &d_hitgroup_records ),
                hitgroup_record_size * RAY_TYPE_COUNT * num_materials
                ));

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * num_materials];
    for( int i = 0; i < num_materials; ++i )
    {
      {
        const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

        OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_hit_group, &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.material = materials[i];
        hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float3*>(state.d_vertices);
        hitgroup_records[sbt_idx].data.normals  = reinterpret_cast<float3*>(state.d_normals);
      }

      {
        const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material

        OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_hit_group, &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.material = materials[i];
        hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float3*>(state.d_vertices);
        hitgroup_records[sbt_idx].data.normals  = reinterpret_cast<float3*>(state.d_normals);

      }
    }

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_hitgroup_records ),
                hitgroup_records,
                hitgroup_record_size * RAY_TYPE_COUNT * num_materials,
                cudaMemcpyHostToDevice
                ) );
    state.sbt.raygenRecord                = d_raygen_record;
    state.sbt.missRecordBase              = d_miss_records;
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase          = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * num_materials;
  }


  /**** INIT PARAMS ****/

  void init_params(RendererState & state,
                   int samples_per_launch,
                   int width,
                   int height,
                   Camera camera)
  {
    state.params.width = width;
    state.params.height = height;
    CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>( &state.params.accum_buffer ),
                state.params.width * state.params.height * sizeof( float4 )
                ));
                
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index     = 0u;

    state.params.handle         = state.d_gas_handler;

    sutil::Camera scamera;
    scamera.setEye(camera.eye);
    scamera.setLookat(camera.look_at);
    scamera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    scamera.setFovY(camera.fov);
    scamera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));

    state.params.eye = scamera.eye();
    scamera.UVWFrame(state.params.U, state.params.V, state.params.W);

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
  }


  /**** CLEAN STATE ****/

  void clean_state(RendererState &state) {
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_miss_group));
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));


    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_normals)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
  }
}