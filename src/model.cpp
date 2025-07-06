//
// Created by progamers on 6/30/25.
//
#include "model.h"

#include <exception>

#include "mesh.h"
#include "shader.h"
#include "stb_image.h"

namespace raw {
model::model(const std::string &path) {
	load_model(path);
}
void model::draw(raw::shader &shader) {
	for (auto &mesh : meshes) {
		mesh.draw(shader);
	}
}

void model::load_model(const std::string &path) {
	Assimp::Importer importer;
	const aiScene	*scene = importer.ReadFile(
		  path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		throw std::runtime_error(std::string(
			std::format("Failed to load model from {}: {}", path, importer.GetErrorString())));
	}
	directory = path.substr(0, path.find_last_of('/'));
	process_node(*scene->mRootNode, *scene);
}

void model::process_node(const aiNode &node, const aiScene &model) {
	for (unsigned int i = 0; i < node.mNumMeshes; ++i) {
		meshes.push_back(process_mesh(*model.mMeshes[node.mMeshes[i]], model));
	}
	for (unsigned int i = 0; i < node.mNumChildren; ++i) {
		process_node(*node.mChildren[i], model);
	}
}

mesh model::process_mesh(const aiMesh &ai_mesh, const aiScene &model) {
	vec<vertex>	 vertices;
	vec<UI>		 indices;
	vec<texture> textures;
	for (unsigned int i = 0; i < ai_mesh.mNumVertices; ++i) {
		vertex v;

		v.pos = glm::vec3(ai_mesh.mVertices[i].x, ai_mesh.mVertices[i].y, ai_mesh.mVertices[i].z);

		if (ai_mesh.HasNormals()) {
			v.normal =
				glm::vec3(ai_mesh.mNormals[i].x, ai_mesh.mNormals[i].y, ai_mesh.mNormals[i].z);
		} else {
			std::cout << "ALARM! MESH '" << ai_mesh.mName.C_Str()
					  << "' HAS NO NORMALS AFTER ASSIMP PROCESSING!" << std::endl;
			v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
		}

		if (ai_mesh.mTextureCoords[0]) {
			v.tex_coords =
				glm::vec2(ai_mesh.mTextureCoords[0][i].x, ai_mesh.mTextureCoords[0][i].y);
		} else {
			v.tex_coords = glm::vec2(0.0f, 0.0f);
		}
		vertices.push_back(v);
	}

	for (unsigned int i = 0; i < ai_mesh.mNumFaces; ++i) {
		const aiFace &face = ai_mesh.mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; ++j) {
			indices.push_back(face.mIndices[j]);
		}
	}

	if (ai_mesh.mMaterialIndex >= 0 && ai_mesh.mMaterialIndex < model.mNumMaterials) {
		const aiMaterial &material = *model.mMaterials[ai_mesh.mMaterialIndex];
		textures = load_material_textures(material, aiTextureType_DIFFUSE, "texture_diffuse");

		auto specular_textures =
			load_material_textures(material, aiTextureType_SPECULAR, "texture_specular");
		textures.insert(textures.end(), specular_textures.begin(), specular_textures.end());
	}
	return {vertices, indices, textures};
}

vec<texture> model::load_material_textures(const aiMaterial &material, aiTextureType type,
										   const std::string &type_name) {
	vec<texture> textures_for_mesh;
	for (unsigned int i = 0; i < material.GetTextureCount(type); ++i) {
		aiString str;
		material.GetTexture(type, i, &str);
		bool skip = false;
		for (size_t j = 0; j < loaded_textures.size(); ++j) {
			if (std::strcmp(loaded_textures[j].path.data(), str.C_Str()) == 0) {
				textures_for_mesh.push_back(loaded_textures[j]);
				skip = true;
				break;
			}
		}
		if (!skip) {
			texture tex;
			tex.id	 = texture_from_file(str.C_Str(), directory);
			tex.type = type_name;
			tex.path = std::string(str.C_Str());
			textures_for_mesh.push_back(tex);
			loaded_textures.push_back(tex);
		}
	}
	return textures_for_mesh;
}

UI model::texture_from_file(const char *path, const std::string &directory, bool gamma) {
	std::string filename = std::string(path);
	filename			 = directory + '/' + filename;

	std::cout << "Attempting to load texture: " << filename << std::endl;

	unsigned int textureID;
	glGenTextures(1, &textureID);

	int			   width, height, nrComponents;
	unsigned char *data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
	if (data) {
		GLenum format;
		if (nrComponents == 1)
			format = GL_RED;
		else if (nrComponents == 3)
			format = GL_RGB;
		else if (nrComponents == 4)
			format = GL_RGBA;

		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		stbi_image_free(data);
	} else {
		std::cout << "Texture failed to load at path: " << path << std::endl;
		stbi_image_free(data);
	}

	return textureID;
}

} // namespace raw