//
// Created by progamers on 6/22/25.
//
#include "mesh.h"
#include "shader.h"

#include <glad/glad.h>

#include <iostream>

namespace raw {

mesh::mesh(vec<raw::vertex>& vcs, vec<UI>& ids, vec<raw::texture>& txs) {
	this->vertices = vcs;
	this->indices  = ids;
	this->textures = txs;
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

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)0);
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

// that's probably slower than just std::to_string, but I also don't care, so I would use that one (just for fun)
void to_char_ptr(char *number, UI& amount_of_digits, UI &number_of_map) {
    constexpr char bits = '0' - 1;
	for (UI num = 0; num < amount_of_digits; ++num) {
		number[num] = bits + char(number_of_map / int(pow(10, num)) % 10);
	}
	++number_of_map;
}

void mesh::draw(raw::shader &shader) {
	UI diffuse_num	= 1;
	UI specular_num = 1;

	for (UI i = 0; i < textures.size(); ++i) {
		glActiveTexture(GL_TEXTURE0 + i);
        UI amount_of_digits = 1;
        if(diffuse_num > 9) {
            amount_of_digits = std::log(diffuse_num);
        }


		char *number = (char *)malloc(sizeof(char) * amount_of_digits);
		if (!number) {
			// START FCKING SCREAMING THAT NO MEMORY EVEN FOR 1-2 BYTES MF (LIKE WTHELL)
			std::cerr << "WHO THE FUCK DESIGNED YOUR PC, OR WTF ARE YOU EVEN USING IT FOR\n";
			std::cerr << "Allocation failed ( " << __LINE__ << ") in file " << __FILE__ << "\n";
			free(number);
			exit(255);
		}
		std::string name = textures[i].type;
		if (name == "texture_diffuse") {
            to_char_ptr(number, amount_of_digits, diffuse_num);
		} else if (name == "texture_specular") {
            to_char_ptr(number, amount_of_digits, specular_num);
		}
		shader.set_float("material." + name + number, i);
        glBindTexture(GL_TEXTURE_2D, textures[i].id);
        free(number);
	}
    glActiveTexture(GL_TEXTURE0);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

} // namespace raw