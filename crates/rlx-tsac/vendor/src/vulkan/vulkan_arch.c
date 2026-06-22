#define _GNU_SOURCE
#include "../../include/tsac.h"
#include "../tsac_codec.h"
#include "../dac_model.h"
#include "vulkan_shaders.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <vulkan/vulkan.h>

#define MAX_SHADERS 16
#define MAX_SETS 16

struct VkBackend {
    void *lib;
    VkInstance instance; VkPhysicalDevice phys_dev; VkDevice device;
    VkQueue queue; VkCommandPool pool; VkCommandBuffer cmd;
    VkFence fence; VkDescriptorPool desc_pool;
    VkDescriptorSetLayout desc_layout; VkDescriptorSet desc_set;
    VkPipelineLayout pipe_layout;
    VkPipeline pipelines[MAX_SHADERS];
    VkShaderModule shader_modules[MAX_SHADERS];
    int n_pipelines;
    int initialized;
};

/* Vulkan function pointers */
#define VK_FUNC(name) static PFN_##name pfn_##name
VK_FUNC(vkCreateInstance); VK_FUNC(vkDestroyInstance);
VK_FUNC(vkEnumeratePhysicalDevices); VK_FUNC(vkGetPhysicalDeviceProperties);
VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FUNC(vkCreateDevice); VK_FUNC(vkDestroyDevice); VK_FUNC(vkGetDeviceQueue);
VK_FUNC(vkCreateCommandPool); VK_FUNC(vkDestroyCommandPool);
VK_FUNC(vkAllocateCommandBuffers); VK_FUNC(vkFreeCommandBuffers);
VK_FUNC(vkCreateFence); VK_FUNC(vkDestroyFence);
VK_FUNC(vkWaitForFences); VK_FUNC(vkResetFences);
VK_FUNC(vkBeginCommandBuffer); VK_FUNC(vkEndCommandBuffer); VK_FUNC(vkQueueSubmit);
VK_FUNC(vkCreateDescriptorPool); VK_FUNC(vkDestroyDescriptorPool);
VK_FUNC(vkCreateShaderModule); VK_FUNC(vkDestroyShaderModule);
VK_FUNC(vkCreatePipelineLayout); VK_FUNC(vkDestroyPipelineLayout);
VK_FUNC(vkCreateComputePipelines); VK_FUNC(vkDestroyPipeline);
VK_FUNC(vkCreateDescriptorSetLayout); VK_FUNC(vkDestroyDescriptorSetLayout);
VK_FUNC(vkAllocateDescriptorSets); VK_FUNC(vkUpdateDescriptorSets);
VK_FUNC(vkCmdBindPipeline); VK_FUNC(vkCmdBindDescriptorSets);
VK_FUNC(vkCmdPushConstants); VK_FUNC(vkCmdDispatch);
VK_FUNC(vkCmdPipelineBarrier);
VK_FUNC(vkCreateBuffer); VK_FUNC(vkDestroyBuffer);
VK_FUNC(vkGetBufferMemoryRequirements); VK_FUNC(vkAllocateMemory);
VK_FUNC(vkFreeMemory); VK_FUNC(vkBindBufferMemory);
VK_FUNC(vkMapMemory); VK_FUNC(vkUnmapMemory);
VK_FUNC(vkFlushMappedMemoryRanges); VK_FUNC(vkInvalidateMappedMemoryRanges);
#undef VK_FUNC

#define LOAD(name) do { \
    pfn_##name = (PFN_##name)dlsym(b->lib, #name); \
    if (!pfn_##name) { fprintf(stderr, "[vk] no " #name "\n"); goto fail; } \
} while(0)

static int create_pipeline(struct VkBackend *b, int idx, const uint8_t *spv, size_t spv_len) {
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv_len;
    smci.pCode = (const uint32_t*)spv;
    if (pfn_vkCreateShaderModule(b->device, &smci, NULL, &b->shader_modules[idx]) != VK_SUCCESS)
        return -1;

    VkPipelineShaderStageCreateInfo ssci = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = b->shader_modules[idx];
    ssci.pName = "main";

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci;
    cpci.layout = b->pipe_layout;
    return pfn_vkCreateComputePipelines(b->device, VK_NULL_HANDLE, 1, &cpci, NULL, &b->pipelines[idx]);
}

int tsac_vk_init(void **priv) {
    if (!priv) return TSAC_ERR_PARAM;
    struct VkBackend *b = calloc(1, sizeof(*b));
    if (!b) return TSAC_ERR_MEMORY;

    b->lib = dlopen("libvulkan.so", RTLD_LAZY|RTLD_LOCAL);
    if (!b->lib) b->lib = dlopen("libvulkan.so.1", RTLD_LAZY|RTLD_LOCAL);
    if (!b->lib) { fprintf(stderr,"[vk] no libvulkan\n"); goto fail; }

    LOAD(vkCreateInstance); LOAD(vkDestroyInstance);
    LOAD(vkEnumeratePhysicalDevices); LOAD(vkGetPhysicalDeviceProperties);
    LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD(vkCreateDevice); LOAD(vkDestroyDevice); LOAD(vkGetDeviceQueue);
    LOAD(vkCreateCommandPool); LOAD(vkDestroyCommandPool);
    LOAD(vkAllocateCommandBuffers); LOAD(vkFreeCommandBuffers);
    LOAD(vkCreateFence); LOAD(vkDestroyFence);
    LOAD(vkWaitForFences); LOAD(vkResetFences);
    LOAD(vkBeginCommandBuffer); LOAD(vkEndCommandBuffer); LOAD(vkQueueSubmit);
    LOAD(vkCreateDescriptorPool); LOAD(vkDestroyDescriptorPool);
    LOAD(vkCreateShaderModule); LOAD(vkDestroyShaderModule);
    LOAD(vkCreatePipelineLayout); LOAD(vkDestroyPipelineLayout);
    LOAD(vkCreateComputePipelines); LOAD(vkDestroyPipeline);
    LOAD(vkCreateDescriptorSetLayout); LOAD(vkDestroyDescriptorSetLayout);
    LOAD(vkAllocateDescriptorSets); LOAD(vkUpdateDescriptorSets);
    LOAD(vkCmdBindPipeline); LOAD(vkCmdBindDescriptorSets);
    LOAD(vkCmdPushConstants); LOAD(vkCmdDispatch);
    LOAD(vkCmdPipelineBarrier);
    LOAD(vkCreateBuffer); LOAD(vkDestroyBuffer);
    LOAD(vkGetBufferMemoryRequirements); LOAD(vkAllocateMemory);
    LOAD(vkFreeMemory); LOAD(vkBindBufferMemory);
    LOAD(vkMapMemory); LOAD(vkUnmapMemory);
    LOAD(vkFlushMappedMemoryRanges); LOAD(vkInvalidateMappedMemoryRanges);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (pfn_vkCreateInstance(&ici,NULL,&b->instance)!=VK_SUCCESS) goto fail;

    uint32_t nd=0; pfn_vkEnumeratePhysicalDevices(b->instance,&nd,NULL);
    VkPhysicalDevice devs[8]; nd=nd>8?8:nd;
    pfn_vkEnumeratePhysicalDevices(b->instance,&nd,devs);
    if(!nd) goto fail; b->phys_dev=devs[0];

    VkPhysicalDeviceProperties props;
    pfn_vkGetPhysicalDeviceProperties(b->phys_dev,&props);
    fprintf(stderr,"[vk] GPU: %s\n", props.deviceName);

    uint32_t nqf=0; pfn_vkGetPhysicalDeviceQueueFamilyProperties(b->phys_dev,&nqf,NULL);
    VkQueueFamilyProperties qfp[16]; nqf=nqf>16?16:nqf;
    pfn_vkGetPhysicalDeviceQueueFamilyProperties(b->phys_dev,&nqf,qfp);
    int qfi=-1;
    for(uint32_t i=0;i<nqf;i++) if(qfp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfi=i;break;}
    if(qfi<0) goto fail;

    float qp=1.0f;
    VkDeviceQueueCreateInfo dqci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex=qfi; dqci.queueCount=1; dqci.pQueuePriorities=&qp;
    VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&dqci;
    if(pfn_vkCreateDevice(b->phys_dev,&dci,NULL,&b->device)!=VK_SUCCESS) goto fail;
    pfn_vkGetDeviceQueue(b->device,qfi,0,&b->queue);

    VkCommandPoolCreateInfo cpci={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex=qfi; cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pfn_vkCreateCommandPool(b->device,&cpci,NULL,&b->pool);

    VkCommandBufferAllocateInfo cbai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=b->pool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount=1;
    pfn_vkAllocateCommandBuffers(b->device,&cbai,&b->cmd);

    VkFenceCreateInfo fci={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    pfn_vkCreateFence(b->device,&fci,NULL,&b->fence);

    VkDescriptorPoolSize dps[1]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,16}};
    VkDescriptorPoolCreateInfo dpci={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=4; dpci.poolSizeCount=1; dpci.pPoolSizes=dps;
    pfn_vkCreateDescriptorPool(b->device,&dpci,NULL,&b->desc_pool);

    /* Descriptor set layout: 4 storage buffers (in, w, bias, out) */
    VkDescriptorSetLayoutBinding bindings[4]={};
    for(int i=0;i<4;i++){
        bindings[i].binding=i; bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount=1; bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount=4; dslci.pBindings=bindings;
    pfn_vkCreateDescriptorSetLayout(b->device,&dslci,NULL,&b->desc_layout);

    VkDescriptorSetAllocateInfo dsai={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=b->desc_pool; dsai.descriptorSetCount=1;
    dsai.pSetLayouts=&b->desc_layout;
    pfn_vkAllocateDescriptorSets(b->device,&dsai,&b->desc_set);

    /* Pipeline layout with push constants */
    VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,128};
    VkPipelineLayoutCreateInfo plci={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&b->desc_layout;
    plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    pfn_vkCreatePipelineLayout(b->device,&plci,NULL,&b->pipe_layout);

    /* Create compute pipelines from embedded SPIR-V */
    b->n_pipelines = 0;
    for(int i=0;i<shader_table_count;i++){
        if(create_pipeline(b,i,shader_table[i].code,shader_table[i].len)==0)
            b->n_pipelines++;
        else fprintf(stderr,"[vk] shader '%s' compile failed\n",shader_table[i].name);
    }
    fprintf(stderr,"[vk] %d/%d pipelines ready\n",b->n_pipelines,shader_table_count);

    b->initialized=1; *priv=b;
    fprintf(stderr,"[vk] Vulkan compute backend ready\n");
    return TSAC_OK;

fail:
    if(b->instance) pfn_vkDestroyInstance(b->instance,NULL);
    if(b->lib) dlclose(b->lib);
    free(b);
    return TSAC_ERR_BACKEND;
}

void tsac_vk_shutdown(void *priv) {
    if(!priv) return;
    struct VkBackend *b = priv;
    for(int i=0;i<b->n_pipelines;i++){
        if(b->pipelines[i]) pfn_vkDestroyPipeline(b->device,b->pipelines[i],NULL);
        if(b->shader_modules[i]) pfn_vkDestroyShaderModule(b->device,b->shader_modules[i],NULL);
    }
    if(b->desc_layout) pfn_vkDestroyDescriptorSetLayout(b->device,b->desc_layout,NULL);
    if(b->pipe_layout) pfn_vkDestroyPipelineLayout(b->device,b->pipe_layout,NULL);
    if(b->desc_pool) pfn_vkDestroyDescriptorPool(b->device,b->desc_pool,NULL);
    if(b->fence) pfn_vkDestroyFence(b->device,b->fence,NULL);
    if(b->pool) pfn_vkDestroyCommandPool(b->device,b->pool,NULL);
    if(b->device) pfn_vkDestroyDevice(b->device,NULL);
    if(b->instance) pfn_vkDestroyInstance(b->instance,NULL);
    if(b->lib) dlclose(b->lib);
    free(b);
}
