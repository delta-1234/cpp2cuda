// Trace the CUDA runtime's actual launch return, including the kernel symbol chosen by the runtime.
#include <cupti.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

namespace {
CUpti_SubscriberHandle subscriber;
void CUPTIAPI on_launch(void*, CUpti_CallbackDomain, CUpti_CallbackId, const void* data) {
    const auto* cb = static_cast<const CUpti_CallbackData*>(data);
    if (cb->callbackSite == CUPTI_API_EXIT && cb->functionReturnValue) {
        std::fprintf(stderr, "CUPTI_LAUNCH %d %s\n",
                     static_cast<int>(*static_cast<const cudaError_t*>(cb->functionReturnValue)),
                     cb->symbolName ? cb->symbolName : "(unknown)");
    }
}
__attribute__((constructor)) void enable_trace() {
    if (cuptiSubscribe(&subscriber, on_launch, nullptr) != CUPTI_SUCCESS ||
        cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) != CUPTI_SUCCESS) {
        std::fputs("could not subscribe to CUDA runtime launches\n", stderr);
        std::abort();
    }
}
}
