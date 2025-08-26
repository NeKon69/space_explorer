//
// Created by progamers on 7/7/25.
//

#ifndef SPACE_EXPLORER_MESH_GENERATOR_H
#define SPACE_EXPLORER_MESH_GENERATOR_H
#include <raw_memory.h>
#include "sphere_generation/fwd.h"
#include <array>
#include <glm/glm.hpp>

#include "cuda_types/buffer.h"
#include "cuda_types/cuda_from_gl_data.h"
#include "helper/helper_macros.h"
#include "rendering/vertex.h"

namespace raw {
    // this class serves as a nice thing to warp around hard things in generating sphere from
    // icosahedron
    class icosahedron_generator {
    private:
        cuda_from_gl_data<raw::vertex> vertices_handle;
        cuda_from_gl_data<UI> indices_handle;
        raw::shared_ptr<cuda_stream> stream;

        UI _vbo;
        UI _ebo;

        cuda_buffer<raw::vertex> vertices_second;
        cuda_buffer<UI> indices_second;
        cuda_buffer<uint32_t> amount_of_triangles;
        cuda_buffer<uint32_t> amount_of_vertices;

        size_t vertices_bytes = 0;
        size_t indices_bytes = 0;

        uint32_t num_vertices_cpu = 12;
        uint32_t num_triangles_cpu = predef::BASIC_AMOUNT_OF_TRIANGLES;

        bool inited = false;

        // Called every time after `generate` function
        void cleanup();

        // Called once when the object is created (or generate function called first time)
        void init(UI vbo, UI ebo);

        // Called every time `generate` function
        void prepare(UI vbo, UI ebo);

    public:
        icosahedron_generator();

        icosahedron_generator(UI vbo, UI ebo, UI steps = predef::BASIC_STEPS);

        void generate(UI vbo, UI ebo, UI steps);

        static constexpr std::array<glm::vec3, 12> generate_icosahedron_vertices();

        static constexpr std::array<UI, 60> generate_icosahedron_indices();

        static constexpr std::pair<std::array<glm::vec3, 12>, std::array<UI, 60> >
        generate_icosahedron_data();
    };
} // namespace raw
#endif // SPACE_EXPLORER_MESH_GENERATOR_H