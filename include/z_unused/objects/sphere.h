//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_SPHERE_H
#define SPACE_EXPLORER_SPHERE_H
#include "../z_unused/object.h"
#include "n_body/n_body_predef.h"
#include "sphere_generation/mesh_generator.h"

namespace raw::z_unused::objects {
class sphere {
private:
	std::vector<UI>							 indices;
	std::vector<float>						 vertices;
	z_unused::object						 obj;
	sphere_generation::icosahedron_data_manager gen;

public:
	explicit sphere(const raw::shared_ptr<raw::rendering::shader::shader> &,
					float radius = sphere_generation::predef::BASIC_RADIUS);

	// Yea yea it would've been more correct to use inheritance here, but for some unknown, very
	// important reason i can't
	void rotate(float degree, const glm::vec3 &axis);

	void move(const glm::vec3 &vec);

	void scale(const glm::vec3 &factor);

	void rotate_around(const float degree, const glm::vec3 &axis,
					   const glm::vec3 &distance_to_object);

	glm::mat4 get_mat() const {
		return obj.get_mat();
	}

	void reset();

	void set_shader(const raw::shared_ptr<raw::rendering::shader::shader> &sh);

	/**
	 * \brief
	 * draw the object
	 * \param reset should matrix reset? defaults to true
	 */
	void draw(decltype(drawing_method::drawing_method) drawing_method = drawing_method::basic,
			  bool									   reset		  = true);
};
} // namespace raw::z_unused::objects
#endif // SPACE_EXPLORER_SPHERE_H