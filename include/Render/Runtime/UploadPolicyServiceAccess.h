#pragma once

#include "Render/Runtime/IUploadPolicyService.h"

namespace rg::runtime {

inline IUploadPolicyService*& UploadPolicyServiceSlot() {
    static IUploadPolicyService* service = nullptr;
    return service;
}

inline void SetActiveUploadPolicyService(IUploadPolicyService* service) {
    UploadPolicyServiceSlot() = service;
}

inline IUploadPolicyService* GetActiveUploadPolicyService() {
    return UploadPolicyServiceSlot();
}

inline void RegisterUploadPolicyClient(IUploadPolicyClient* client) {
    if (!client) {
        return;
    }

    if (auto* service = GetActiveUploadPolicyService()) {
        service->RegisterClient(client);
    }
}

inline void UnregisterUploadPolicyClient(IUploadPolicyClient* client) {
    if (!client) {
        return;
    }

    if (auto* service = GetActiveUploadPolicyService()) {
        service->UnregisterClient(client);
    }
}

inline void BeginUploadPolicyFrame() {
    if (auto* service = GetActiveUploadPolicyService()) {
        service->BeginFrame();
    }
}

inline void FlushUploadPolicies() {
    if (auto* service = GetActiveUploadPolicyService()) {
        service->FlushAll();
    }
}

}
