# Vulkan
## Swapchain

Vulkan doesn't have a concept of default framebuffer to draw on. It instead
requires you to create a number of framebuffer images and bind them into a
swapchain. This gives the developer a greater control over the presentation of
images while implementing buffering.

Swapchain is simply a queue of framebuffer images. Developer can request an
unused framebuffer image, submit a draw queue to the GPU to draw on the
framebuffer or send submit a presentation queue to the Window Integration System
to present the framebuffer to the screen. It's a highly configurable system
which allows most of it internals to be set by the developer, like framebuffer
count, surface image formats and presentation modes.

The most important aspect of a swapchain is its presentation mode. This is the
option that is responsible for how the images are returned to the system and in
which order they are presented. They must be chosen based on the needs of the
application. One can wish to use a mode that prevents screen tearing or another
mode that reduces the input lag. Developer can implement double buffering,
triple buffering or no buffering at all, using these modes. It can even set the
buffer count to a higher number like 256, which some GPUs support.

There are 4 presentation modes in core Vulkan specifications:

- `VK_PRESENT_MODE_IMMEDIATE_KHR`: The framebuffer images are presented to the
screen immediately which often causes screen tearing but results in very little
delay. It can cause GPU to draw in very high frame rates even if the screen
doesn't support such high refresh rates. This can result in increased power
consumption which is not desired in mobile devices.

- `VK_PRESENT_MODE_FIFO_KHR`: This mode contains a front and at least one back
buffer, which the system swaps at vertical blanks. The rendering is done to a
framebuffer at the back which doesn't cause screen tearing because the front
buffer isn't updated. This is one of the best modes for the mobile and low power
devices as it waits for the new refresh to draw a new frame. Which prevents GPU
from drawing a frame that wouldn't be presented and decreases the power
consumption. This may cause noticable delay if the supported refresh rate isn't
high enough because it doesn't render more up-to-date frames unlike immediate
mode. If the rendering is not finished until the next vertical blank, the swap
doesn't occur and the system waits for the next refresh. This method is similar
to double buffering.

- `VK_PRESENT_MODE_FIFO_RELAXED_KHR`: The only difference between this mode and
the previous first in first out mode is that this one doesn't wait for the next
refresh if the rendering isn't finished by the time of vertical blank. It then
presents the finished image right away without waiting for the next refresh.
This can cause tearing but results in lower delays compared to the previous one.

- `VK_PRESENT_MODE_MAILBOX_KHR`: This mode prevents the screen tearing while
allowing more up-to-date images than FIFO mode. Rendering is done to a back
buffer but instead of waiting for vertical blank, application can keep on
drawing to another back buffer which then replaces the old back buffer in the
presentation queue if the drawing is finished before the next refresh. The most
up-to-date back buffer is presented to the screen at vertical blank. This
prevents screen tearing since no drawing is to the front buffer and delays are
still comparable to immediate mode. Unfortunately power consumption is higher
than FIFO but this mode is must be used if it isn't a concern.

There are more presentation modes added to Vulkan with extensions, for more
detailed information, you can check [the full specifications](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPresentModeKHR.html).

Both screen tearing and input lag can cause motion sickness while using VR,
therefore using mailbox is the best choice.

## Pipelines

Older graphical APIs like OpenGL and DirectX support changing pipeline
functionality on the fly but almost all pipeline functionality of Vulkan is
immutable. This means that developer must compile different pipelines in advance
to use for different situations, for example when it is necessary to bind
another shader. This allows the driver to make further optimizations to the
pipeline since all the properties are knows beforehand. Application then can
switch between pipeline when needed. Since it's a very expensive task to
create and compile new pipelines, it's a no-go to create them on the fly. Even
when it's absolutely necessary to create one, it is best not to create it in the
main thread.

Vulkan allows some pipeline functionality to be modified, hovewer. Using [Vulkan's Dynamic State](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkDynamicState.html)
one can modify some aspects of the pipeline. While it is limited to viewport,
scissor and some reference values like blend constants or stencil masks in core
Vulkan, extensions add back most of the functionality on supported hardwares.
Here we make use of dynamic state for stencil values. Otherwise we need to
compile a separate pipeline per portal view.

### Bindless Architecture

[GPU Driven Rendering](https://vkguide.dev/docs/gpudriven/gpu_driven_engines/)

## SPIR-V Modules
## Queues and Command Buffers
### Pre-recording Command Buffers

Since the cost of recording command buffers isn't trivial, it makes sense for
static scenes, to record command buffers while compiling pipelines and submit
these commands every frame. But this may not always possible due to the dynamic
nature of some scenes. For portal rendering, since it's not known beforehand
which portals are visible and which aren't, it's almost impossible to record and
optimal command buffer once and use it later. So it's necessary to do a
visiblity check before every frame and record command buffers accordingly. This
allows application to not waste time on out-of-screen portals and focus only the
visible ones.

### Dynamically Recording per Frame
## Multithreading and Syncronization

# Virtual Reality
## Motion Sickness
## Oculus Asyncronous Spacewarp

Oculus has a technology named [Asyncronous Spacewarp](https://www.oculus.com/blog/introducing-asw-2-point-0-better-accuracy-lower-latency/)
that uses temporal properties of the scene and machine learning techniques to
approximate the next frame if the GPU can't keep up with the particular frame
rates (namely 90 frames per second). It creates a new frame based on the
previous one instead of rendering it. This makes the application look smother
but since it's not perfect, there are reports that it makes some people suffer
from motion sickness when activated. So it's important to always stay above the
frame rate limits.

## Multiview Extension

Virtual Reality applications contain 2 different viewports with slightly
different camera positions to simulate human eyes and trick the brain into
thinking that it's looking at a 3D world instead of computer screen. Even
though rendered images are a bit different from each other, they have a lot in
common and could share most of the pipeline stages if given chance.

This is where [Vulkan's Multiview Extension](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_multiview.html)
come in. It allows the creation of single pipeline with 2 different viewports so
that 2 views share the same pipeline as much as possible. This has a slight
performance benefit over 2 separate pipelines since it eliminates the need of
pipeline switching while rendering. Also provides some constant referans values
that are accessible in the shaders to make distinction between the views.

# Portals
## Offscreen Rendering (Render to Texture)
## Stencil Buffer
## Oblique Frustum
## Recursion
## Cell-Portal Graphs
## Infinite Space Limitations
## Compute Shaders

# Comparisons
