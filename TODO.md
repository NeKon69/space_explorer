### 🔥 **High Priority: Core Refactoring**

*These tasks are crucial for stability and need to be addressed now.*

* `⚙️ [ 📝 Planned ]` - Fix n-body by dividing single kernel into 2 separate ones, "kick-drift" and "kick" to properly
  work on more than 256 objects
* `🧐 [ 📝 Planned ]` - Check if UV coordinate system actually works and if it has "edges"
* `🛠️ [ 🚧 In Progress]` - (Optional, either do it now or never) Refactor interaction_system into 2 classes and make it
  more reliable

---

### 🚀 **High Priority: New Features**

*Key features that need to be implemented for the next milestone.*

* `🎨 [ ⏳ On Hold]` - Finish texture streaming/generations
* `🌊 [ 📝 Planned]` - Introduce CUDA/OpenCL streams to each part of project (n-body, sphere tesselation, texture
  generation) to prevent chaos of randomly creating/deleting threads locally.
* `🎛️ [ 📝 Planned]` - Introduce manager for GPU streams

---

### 🗓️ **Future Goals: Architecture & Long-Term Refactoring**

*Important structural changes to be addressed after the current priorities are completed.*

* `🏛️ [ 🚧 In Progress]` - Make base classes for GPU-backend from which other classes will be inherited based on chosen
  backend (OpenCL or CUDA)
* `🧩 [ 📝 Planned]` - Refactor "playing_state" class to be more modular, introduce "event_manager", "input_handler", and
  other stuff

---

### ✅ **Completed Tasks**

*A log of what's already been done.*

* `[ ✅ Done ]` - ~~Divide kernel for sphere tesselation into 2 kernels, one for vertices and one for another TBN data~~
* `[ ✅ Done ]` - ~~Split "icosahedron_generator" into two classes, one for managing resources second for launching
  kernels~~
* `[ ✅ Done ]` - ~~Update basic vertices/indices generators to support normal mapping~~
* `[ ✅ Done ]` - ~~Negate vertices duplicating in the "sphere tessellation" process by using several kernels, this need
  to be done to prevent bullshit UV/TBN fuckups.~~
* `[ ✅ Done ]` - ~~Refactor cuda_from_gl_data class to be inherited from resource class and make it another class in "
  from_gl" folder~~
* `[ ✅ Done ]` - ~~Refactor "buffer.h" to be more robust, right now it's bloated with trash~~
