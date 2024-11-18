// This file is created by spikingjelly.clock_driven.neuron_kernel.save_cuda_codes.

// MultiStepIFNodePTT

// MultiStepIFNodePTT fptt ATan, hard_reset=True, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + x_seq[t];
                        if (h_seq[t] >= v_threshold)
                
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt ATan, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt ATan, hard_reset=False, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + x_seq[t];
                        if (h_seq[t] >= v_threshold)
                
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt ATan, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + x_seq[t];
                        if (h_seq[t] >= v_threshold)
                
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + x_seq[t];
                        if (h_seq[t] >= v_threshold)
                
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + x_seq[t];
                        if (h_seq[t] >= v_threshold)
                
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32

                extern "C" __global__
                void IFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = v_v_seq[t] + x_seq[t];
                        if (h_seq[t] >= v_threshold)
                
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void IFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                    grad_x_seq[t] = grad_h;
                    }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                        h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void IFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;
                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                        
                        grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = grad_h;
                        }
                grad_v_last[index] = grad_x_seq[index];
                }
                }
                
// MultiStepLIFNodePTT

// MultiStepLIFNodePTT fptt ATan, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt ATan, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32

                extern "C" __global__
                void LIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void LIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
                float* grad_x_seq, float* grad_v_last,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                }
                
// MultiStepLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void LIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
                half2* grad_x_seq, half2* grad_v_last,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    }
                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                }
                
// MultiStepParametricLIFNodePTT

// MultiStepParametricLIFNodePTT fptt ATan, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const float sg_ATan_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * 2.0f * over_th;
            				const float grad_s_to_h = 2.0f / 2.0f / (1.0f + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x);
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT fptt ATan, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT bptt ATan, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
            				const half2 sg_ATan_alpha =  __float2half2_rn(2.0f);
            				const half2 sg_ATan_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), sg_ATan_alpha), over_th);
            				const half2 grad_s_to_h = __h2div(__h2div(sg_ATan_alpha, __float2half2_rn(2.0f)), __hfma2(sg_ATan_M_PI_2__alpha__x, sg_ATan_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.ATan.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const float sg_Sigmoid_sigmoid_ax = 1.0f / (1.0f + expf(- 1.0f * over_th));
            				const float grad_s_to_h = (1.0f - sg_Sigmoid_sigmoid_ax) * sg_Sigmoid_sigmoid_ax * 1.0f;
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT fptt Sigmoid, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT bptt Sigmoid, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
            				const half2 sg_Sigmoid_alpha = __float2half2_rn(1.0f);
            				const half2 sg_Sigmoid_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2(sg_Sigmoid_alpha, over_th))), __float2half2_rn(1.0f)));
            				const half2 grad_s_to_h = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_sigmoid_ax), sg_Sigmoid_alpha);
            
            				// end: spikingjelly.clock_driven.surrogate.Sigmoid.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t];
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, const float & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_hardReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=True, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_hardReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, const half & v_reset,
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                        const half2 v_reset_half2 = __half2half2(v_reset);
                    

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32

                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp32(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
                const float & reciprocal_tau, 
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                    
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }

                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=True

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f;
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp32, detach_reset=False

                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp32(
                const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
                float* grad_x_seq, float* grad_v_last, float* grad_reciprocal_tau,
                const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
                const float & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ float sdata[1024];
                    if (index < neuron_num)
                    {   
                        float grad_h = 0.0f;  // grad_h will be used recursively
                        sdata[threadIdx.x] = 0.0f;
                        for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                        {
                            const int t = index + mem_offset;
                            const float over_th = h_seq[t] - v_threshold;
                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const float sg_PiecewiseLeakyReLU_x_abs = fabsf(over_th);
            float grad_s_to_h;
            if (sg_PiecewiseLeakyReLU_x_abs > 1.0f)
            {
                grad_s_to_h = 0.01f;
            }
            else
            {
                grad_s_to_h = 1.0f;
            }
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                        
                    grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                    // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                    }
                grad_v_last[index] = grad_x_seq[index] * one_sub_reciprocal_tau;
                }
                else
                {
                    sdata[threadIdx.x] = 0.0f;
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = sdata[0];
                }
                }
                
// MultiStepParametricLIFNodePTT fptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_fptt_softReset_fp16(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
                const half & reciprocal_tau, 
                const half & v_threshold, 
                const int & neuron_num, const int & numel) 
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;
                if (index < stride)
                {
                    const int numel_2 = numel >> 1;
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                
                    for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                    {
                        const int t = index + mem_offset;
                
                        h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                        spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                        v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                    
                    }
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=True

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset_detachReset_fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __float2half2_rn(1.0f);
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                
// MultiStepParametricLIFNodePTT bptt PiecewiseLeakyReLU, hard_reset=False, dtype=fp16, detach_reset=False

                #include <cuda_fp16.h>
                extern "C" __global__
                void ParametricLIFNode_bptt_softReset__fp16(
                const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
                half2* grad_x_seq, half2* grad_v_last,  half* grad_reciprocal_tau,
                const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
                const half & v_threshold, 
                const int & neuron_num, const int & numel)
                
                {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = neuron_num >> 1;

                __shared__ half2 sdata[1024];
                if (index < stride)
                {   
                    const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                    const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                    const half2 v_threshold_half2 = __half2half2(v_threshold);
                

                    half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                    for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                    {
                        const int t = index + mem_offset;

                        const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

                
            				// start: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
            				const half2 sg_PiecewiseLeakyReLU_x_abs = __habs2(over_th);
            				const half2 sg_PiecewiseLeakyReLU_x_abs_ge_w = __hge2(sg_PiecewiseLeakyReLU_x_abs, __float2half2_rn(1.0f));
            				half2 grad_s_to_h = __hadd2(__hmul2(__float2half2_rn(0.01f),  sg_PiecewiseLeakyReLU_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), sg_PiecewiseLeakyReLU_x_abs_ge_w), __float2half2_rn(1.0f)));
            
            				// end: spikingjelly.clock_driven.surrogate.PiecewiseLeakyReLU.cuda_code
        
                        const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                                                
                        grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        sdata[threadIdx.x] = __hadd2(__h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2), sdata[threadIdx.x]);
                    }

                grad_v_last[index] = __hmul2(grad_x_seq[index], one_sub_reciprocal_tau_half2);
                }
                else
                {
                    sdata[threadIdx.x] = __float2half2_rn(0.0f);
                }
                int threadx = blockDim.x;
                #pragma unroll
                for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
                {
                // Synchronize all thread before next loop
                __syncthreads();
                if (threadIdx.x < stride)
                {
                  sdata[threadIdx.x] = __hadd2(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
                }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                grad_reciprocal_tau[0] = __hadd(__low2half(sdata[0]), __high2half(sdata[0]));
                }
                }
                