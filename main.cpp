#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <vector>

std::vector<char> ReadFile(const std::string& filename) {
    std::ifstream fin(filename, std::ios::ate | std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("failed to open file " + filename);
    }
    size_t size = fin.tellg();
    std::vector<char> result(size);
    fin.seekg(0);
    fin.read(result.data(), size);
    fin.close();
    return result;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

class TApp {
public:
    void Run() {
        InitWindow();
        InitVulkan();
        Loop();
        Cleanup();
    }

private:
    void InitWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        Window_ = glfwCreateWindow(WIDTH, HEIGHT, "vulkan", nullptr, nullptr);
    }

    void InitVulkan() {
        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateGraphicsPipeline();
        CreateFramebuffers();
        CreateCommandPool();
        CreateCommandBuffer();
        CreateSyncObjects();
    }

    void CreateSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        bool good = vkCreateSemaphore(Device_, &semaphoreInfo, nullptr, &ImageAvailableSemaphore_) == VK_SUCCESS;
        good &= vkCreateSemaphore(Device_, &semaphoreInfo, nullptr, &RenderFinishedSemaphore_) == VK_SUCCESS;
        good &= vkCreateFence(Device_, &fenceInfo, nullptr, &InFlightFence_) == VK_SUCCESS;
        if (!good) {
            throw std::runtime_error("failed to create sync primitives");
        }
    }

    void CreateCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = CommandPool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        if (vkAllocateCommandBuffers(Device_, &allocInfo, &CommandBuffer_) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers");
        }
    }

    void CreateCommandPool() {
        auto indices = FindQueueFamilies(PhysicalDevice_);
        VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = indices.GraphicsFamily.value(),
        };

        if (vkCreateCommandPool(Device_, &poolInfo, nullptr, &CommandPool_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool");
        }
    }

    void CreateFramebuffers() {
        SwapChainFramebuffers_.resize(SwapChainImageViews_.size());
        for (uint32_t i = 0; i < SwapChainImageViews_.size(); ++i) {
            VkFramebufferCreateInfo framebufferInfo{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = RenderPass_,
                .attachmentCount = 1,
                .pAttachments = &SwapChainImageViews_.at(i),
                .width = SwapChainExtent_.width,
                .height = SwapChainExtent_.height,
                .layers = 1,
            };

            if (vkCreateFramebuffer(Device_, &framebufferInfo, nullptr, &SwapChainFramebuffers_.at(i)) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer");
            }
        }
    }

    void CreateRenderPass() {
        VkAttachmentDescription colorAttachment{
            .format = SwapChainFormat_,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        VkAttachmentReference colorAttachmentRef{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,

        };

        VkSubpassDependency dep{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        VkRenderPassCreateInfo renderPassInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dep,
        };

        if (vkCreateRenderPass(Device_, &renderPassInfo, nullptr, &RenderPass_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass");
        }
    }


    void CreateGraphicsPipeline() {
        auto vertShader = ReadFile("./vert.spv");
        auto fragShader = ReadFile("./frag.spv");

        auto vertModule = CreateShaderModule(vertShader);
        auto fragModule = CreateShaderModule(fragShader);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertModule,
            .pName = "main",
        };

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragModule,
            .pName = "main",
        };

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 0,
            .vertexAttributeDescriptionCount = 0,
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = (float) SwapChainExtent_.width,
            .height = (float) SwapChainExtent_.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = SwapChainExtent_,
        };

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = (uint32_t) dynamicStates.size(),
            .pDynamicStates = dynamicStates.data(),
        };

        VkPipelineViewportStateCreateInfo viewportState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        VkPipelineRasterizationStateCreateInfo rasterizer{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };

        VkPipelineMultisampleStateCreateInfo multisampling{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };

        VkPipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };

        VkPipelineColorBlendStateCreateInfo colorBlending{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        };

        if (vkCreatePipelineLayout(Device_, &pipelineLayoutInfo, nullptr, &PipelineLayout_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = nullptr,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = PipelineLayout_,
            .renderPass = RenderPass_,
            .subpass = 0,
        };

        if (vkCreateGraphicsPipelines(Device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &GraphicsPipeline_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline");
        }

        vkDestroyShaderModule(Device_, fragModule, nullptr);
        vkDestroyShaderModule(Device_, vertModule, nullptr);
    }

    VkShaderModule CreateShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode = (const uint32_t*) code.data(),
        };
        VkShaderModule result;
        if (vkCreateShaderModule(Device_, &createInfo, nullptr, &result) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module");
        }
        return result;
    }

    void CreateImageViews() {
        SwapChainImageViews_.resize(SwapChainImages_.size());

        for (uint32_t i = 0; i < SwapChainImages_.size(); ++i) {
            VkImageViewCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = SwapChainImages_.at(i),
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = SwapChainFormat_,
                .components = {
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            if (vkCreateImageView(Device_, &createInfo, nullptr, &SwapChainImageViews_.at(i)) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image view");
            }
        }
    }

    void CreateSwapChain() {
        TSwapChainSupportDetails support = QuerySwapChainSupport(PhysicalDevice_);

        auto format = ChooseSwapSurfaceFormat(support.Formats);
        auto presentMode = ChooseSwapPresentMode(support.PresentModes);
        auto extent = ChooseSwapExtent(support.Capabilities);

        auto imgCount = support.Capabilities.minImageCount + 1;
        if (support.Capabilities.maxImageCount > 0) {
            imgCount = std::min(imgCount, support.Capabilities.maxImageCount);
        }

        VkSwapchainCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = Surface_,
            .minImageCount = imgCount,
            .imageFormat = format.format,
            .imageColorSpace = format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = support.Capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };
        auto indices = FindQueueFamilies(PhysicalDevice_);
        uint32_t queueFamilyIndices[] = {indices.GraphicsFamily.value(), indices.PresentFamily.value()};
        if (indices.GraphicsFamily != indices.PresentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        if (vkCreateSwapchainKHR(Device_, &createInfo, nullptr, &SwapChain_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain");
        }

        vkGetSwapchainImagesKHR(Device_, SwapChain_, &imgCount, nullptr);
        SwapChainImages_.resize(imgCount);
        vkGetSwapchainImagesKHR(Device_, SwapChain_, &imgCount, SwapChainImages_.data());

        SwapChainFormat_ = format.format;
        SwapChainExtent_ = extent;
    }

    void CreateSurface() {
        if (glfwCreateWindowSurface(Instance_, Window_, nullptr, &Surface_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create surface");
        }
    }

    void CreateLogicalDevice() {
        auto queueFamilies = FindQueueFamilies(PhysicalDevice_);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            queueFamilies.GraphicsFamily.value(),
            queueFamilies.PresentFamily.value(),
        };

        float queuePriority = 1.0;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = queueFamily,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority,
            };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures physDeviceFeatures{};

        VkDeviceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = (uint32_t) queueCreateInfos.size(),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size()),
            .ppEnabledExtensionNames = DEVICE_EXTENSIONS.data(),
            .pEnabledFeatures = &physDeviceFeatures,
        };

        if (vkCreateDevice(PhysicalDevice_, &createInfo, nullptr, &Device_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create device");
        }

        vkGetDeviceQueue(Device_, queueFamilies.GraphicsFamily.value(), 0, &GraphicsQueue_);
        vkGetDeviceQueue(Device_, queueFamilies.PresentFamily.value(), 0, &PresentQueue_);
    }

    void PickPhysDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(Instance_, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("found no GPUs with Vulkan support");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(Instance_, &deviceCount, devices.data());
        for (const auto& device : devices) {
            if (IsDeviceSuitable(device)) {
                VkPhysicalDeviceProperties properties;
                vkGetPhysicalDeviceProperties(device, &properties);
                std::cerr << "Using device " << properties.deviceName << std::endl;
                PhysicalDevice_ = device;
                break;
            }
        }
        if (PhysicalDevice_ == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU");
        }
    }

    struct TSwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR Capabilities;
        std::vector<VkSurfaceFormatKHR> Formats;
        std::vector<VkPresentModeKHR> PresentModes;
    };

    TSwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) {
        TSwapChainSupportDetails result;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, Surface_, &result.Capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, Surface_, &formatCount, nullptr);
        result.Formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, Surface_, &formatCount, result.Formats.data());

        uint32_t presentModesCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, Surface_, &presentModesCount, nullptr);
        result.PresentModes.resize(formatCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, Surface_, &presentModesCount, result.PresentModes.data());

        return result;
    }

    VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
        return formats.at(0);
    }

    VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& modes) {
        if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()) {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        int w, h;
        glfwGetFramebufferSize(Window_, &w, &h);

        VkExtent2D extent{.width = (uint32_t) w, .height = (uint32_t) h};
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return extent;
    }

    struct TQueueFamilyIndices {
        std::optional<uint32_t> GraphicsFamily;
        std::optional<uint32_t> PresentFamily;

        bool IsComplete() {
            return GraphicsFamily.has_value() && PresentFamily.has_value();
        }
    };

    TQueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
        TQueueFamilyIndices result;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (queueFamilies.at(i).queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                result.GraphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, Surface_, &presentSupport);
            if (presentSupport) {
                result.PresentFamily = i;
            }
        }

        return result;
    }

    bool IsDeviceSuitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            return false;
        }

        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(device, &features);
        if (!features.geometryShader) {
            return false;
        }

        if (!FindQueueFamilies(device).IsComplete()) {
            return false;
        }

        if (!CheckDeviceExtensionSupport(device)) {
            return false;
        }

        auto swapChainSupport = QuerySwapChainSupport(device);
        if (swapChainSupport.PresentModes.empty() || swapChainSupport.Formats.empty()) {
            return false;
        }

        return true;
    }

    bool CheckDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording to command buffer");
        }

        VkClearValue clearColor= {{{0.0f, 0.0f, 0.0f, 1.0f}}};

        VkRenderPassBeginInfo renderPassInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = RenderPass_,
            .framebuffer = SwapChainFramebuffers_.at(imageIndex),
            .renderArea = {
                .offset = {0, 0},
                .extent = SwapChainExtent_,
            },
            .clearValueCount = 1,
            .pClearValues = &clearColor,
        };

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipeline_);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(SwapChainExtent_.width);
        viewport.height = static_cast<float>(SwapChainExtent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = SwapChainExtent_;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer");
        }
    }

    void Loop() {
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t frameCnt = 0;
        while (!glfwWindowShouldClose(Window_)) {
            glfwPollEvents();
            DrawFrame();
            ++frameCnt;
        }
        auto finish = std::chrono::high_resolution_clock::now();
        std::cerr << (double) frameCnt / std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 1000000 << std::endl;
    }

    void DrawFrame() {
        vkWaitForFences(Device_, 1, &InFlightFence_, VK_TRUE, UINT64_MAX);
        vkResetFences(Device_, 1, &InFlightFence_);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(Device_, SwapChain_, UINT64_MAX, ImageAvailableSemaphore_, VK_NULL_HANDLE, &imageIndex);
        vkResetCommandBuffer(CommandBuffer_, 0);
        RecordCommandBuffer(CommandBuffer_, imageIndex);

        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &ImageAvailableSemaphore_,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &CommandBuffer_,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &RenderFinishedSemaphore_,
        };

        if (vkQueueSubmit(GraphicsQueue_, 1, &submitInfo, InFlightFence_) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit to queue");
        }

        VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &RenderFinishedSemaphore_,
            .swapchainCount = 1,
            .pSwapchains = &SwapChain_,
            .pImageIndices = &imageIndex,
        };
        vkQueuePresentKHR(PresentQueue_, &presentInfo);
    }

    void Cleanup() {
        vkDeviceWaitIdle(Device_);
        vkDestroySemaphore(Device_, ImageAvailableSemaphore_, nullptr);
        vkDestroySemaphore(Device_, RenderFinishedSemaphore_, nullptr);
        vkDestroyFence(Device_, InFlightFence_, nullptr);
        vkDestroyCommandPool(Device_, CommandPool_, nullptr);
        for (auto framebuffer : SwapChainFramebuffers_) {
            vkDestroyFramebuffer(Device_, framebuffer, nullptr);
        }
        vkDestroyPipeline(Device_, GraphicsPipeline_, nullptr);
        vkDestroyPipelineLayout(Device_, PipelineLayout_, nullptr);
        vkDestroyRenderPass(Device_, RenderPass_, nullptr);
        for (auto view : SwapChainImageViews_) {
            vkDestroyImageView(Device_, view, nullptr);
        }
        vkDestroySwapchainKHR(Device_, SwapChain_, nullptr);
        vkDestroyDevice(Device_, nullptr);
        if (ENABLE_VALIDATION_LAYERS) {
            DestroyDebugUtilsMessengerEXT(Instance_, DebugMessenger_, nullptr);
        }
        vkDestroySurfaceKHR(Instance_, Surface_, nullptr);
        vkDestroyInstance(Instance_, nullptr);
        glfwDestroyWindow(Window_);
        glfwTerminate();
    }

    void CreateInstance() {
        if (ENABLE_VALIDATION_LAYERS && !CheckValidationLayerSupport()) {
            throw std::runtime_error("validation layers unsupported");
        }

        VkApplicationInfo appInfo {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "no engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_0,
        };

        auto extensions = GetRequiredExtensions();
        VkInstanceCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = 0,
            .enabledExtensionCount = (uint32_t) extensions.size(),
            .ppEnabledExtensionNames = extensions.data(),
        };
        if (ENABLE_VALIDATION_LAYERS) {
            createInfo.enabledLayerCount = (uint32_t) VALIDATION_LAYERS.size();
            createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }

        if (vkCreateInstance(&createInfo, nullptr, &Instance_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance");
        }
    }

    void SetupDebugMessenger() {
        if (!ENABLE_VALIDATION_LAYERS) {
            return;
        }
        VkDebugUtilsMessengerCreateInfoEXT createInfo {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
            .pUserData = nullptr,
        };
        if (CreateDebugUtilsMessengerEXT(Instance_, &createInfo, nullptr, &DebugMessenger_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create debug messenger");
        }
    }

    bool CheckValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layer : VALIDATION_LAYERS) {
            bool found = false;
            for (const auto& layerProps : availableLayers) {
                if (strcmp(layer, layerProps.layerName) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }

        return true;
    }

    std::vector<const char*> GetRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> result(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (ENABLE_VALIDATION_LAYERS) {
            result.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return result;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* udata
    ) {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            std::cerr << "validation layer: " << data->pMessage << std::endl;
        }
        return VK_FALSE;
    }

private:
    static constexpr uint32_t WIDTH = 800;
    static constexpr uint32_t HEIGHT = 600;

    const std::vector<const char*> VALIDATION_LAYERS = {
        "VK_LAYER_KHRONOS_validation"
    };
    const std::vector<const char*> DEVICE_EXTENSIONS = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

#ifdef NDEBUG
    const bool ENABLE_VALIDATION_LAYERS = false;
#else
    const bool ENABLE_VALIDATION_LAYERS = true;
#endif

    GLFWwindow* Window_;
    VkInstance Instance_;
    VkDevice Device_;
    VkQueue GraphicsQueue_;
    VkQueue PresentQueue_;
    VkDebugUtilsMessengerEXT DebugMessenger_;
    VkSurfaceKHR Surface_;
    VkPhysicalDevice PhysicalDevice_ = VK_NULL_HANDLE;

    VkSwapchainKHR SwapChain_;
    VkFormat SwapChainFormat_;
    VkExtent2D SwapChainExtent_;
    std::vector<VkImage> SwapChainImages_;
    std::vector<VkImageView> SwapChainImageViews_;

    VkRenderPass RenderPass_;
    VkPipelineLayout PipelineLayout_;
    VkPipeline GraphicsPipeline_;

    std::vector<VkFramebuffer> SwapChainFramebuffers_;

    VkCommandPool CommandPool_;
    VkCommandBuffer CommandBuffer_;

    VkSemaphore ImageAvailableSemaphore_;
    VkSemaphore RenderFinishedSemaphore_;
    VkFence InFlightFence_;
};

int main() {
    TApp app;

    try {
        app.Run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

