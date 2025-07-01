//
// Created by progamers on 6/22/25.
//
#include "mesh.h"

#include <glad/glad.h>

#include <iostream>

#include "shader.h"

namespace raw {

mesh::mesh(const vec<raw::vertex> &vcs, const vec<UI> &ids, const vec<raw::texture> &txs) {
	this->vertices = vcs;
	this->indices  = ids;
	this->textures = txs;
	setup_mash();
}

void mesh::setup_mash() {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertex), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0],
				 GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), nullptr);
	glEnableVertexAttribArray(0);
	// vertex normals

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex),
						  (void *)offsetof(vertex, normal));
	glEnableVertexAttribArray(1);
	// vertex texture coords

	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex),
						  (void *)offsetof(vertex, tex_coords));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}

// that's probably slower than just std::to_string, but I also don't care, so I would use that one
// (just for fun)
void to_char_ptr(char *number, const UI &amount_of_digits, UI &number_of_map) {
	constexpr char bits = '0';
	for (UI num = 0; num < amount_of_digits; ++num) {
		number[num] = bits + char(number_of_map / int(pow(10, num)) % 10);
	}
	++number_of_map;
}
    void mesh::draw(raw::shader &shader) {
        shader.use();
        shader.set_int("obj_mat.diffuse_map", 0);
        shader.set_int("obj_mat.specular_map", 1);

        bool has_diffuse = false;
        bool has_specular = false;

        for (const auto& texture : textures) {
            if (texture.type == "texture_diffuse") {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texture.id);
                has_diffuse = true;
            } else if (texture.type == "texture_specular") {
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, texture.id);
                has_specular = true;
            }
        }
        if (!has_diffuse) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        if (!has_specular) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);
        glActiveTexture(GL_TEXTURE0);
    }

} // namespace raw