//
// Created by progamers on 6/30/25.
//

#ifndef SPACE_EXPLORER_MODEL_H
#define SPACE_EXPLORER_MODEL_H

#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>

#include "mesh.h"
#include "shader.h"

namespace raw {
class model {
private:
	vec<mesh>	 meshes;
	vec<texture> loaded_textures;
	std::string	 directory;
	void		 load_model(const std::string& path);
	void		 process_node(const aiNode& node, const aiScene& model);
	mesh		 process_mesh(const aiMesh& mesh, const aiScene& model);
	vec<texture> load_material_textures(const aiMaterial& material, aiTextureType type,
										const std::string& type_name);
	UI texture_from_file(const char* path, const std::string& directory, bool gamma = false);

public:
	model() = delete;
	model(const std::string& path);
	void draw(shader& shader);
};
} // namespace raw

#endif // SPACE_EXPLORER_MODEL_H
