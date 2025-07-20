//
// Created by progamers on 7/18/25.
//

#ifndef SPACE_EXPLORER_SPHERE_H
#define SPACE_EXPLORER_SPHERE_H
#include "object.h"
#include "sphere_generation/mesh_generator.h"

namespace raw {
namespace predef {
static const UI MAXIMUM_AMOUNT_OF_INDICES =
	BASIC_AMOUNT_OF_TRIANGLES * static_cast<UI>(std::pow(4, MAX_STEPS)) + 2;
// It's actually 10, but I'll try and see if my algorithm fails on ten, for now, lets keep it at 20
static const UI MAXIMUM_AMOUNT_OF_VERTICES =
	BASIC_AMOUNT_OF_TRIANGLES * static_cast<UI>(std::pow(4, MAX_STEPS));
} // namespace predef
class sphere {
private:
	std::vector<UI>		  indices;
	std::vector<float>	  vertices;
	raw::object			  obj;
	icosahedron_generator gen;

public:
	explicit sphere(const raw::shared_ptr<raw::shader>&, float radius = predef::BASIC_RADIUS);
	// Yea yea it would've been more correct to use inheritance here, but for some unknown, very
	// important reason i can't
	void	  rotate(float degree, const glm::vec3& axis);
	void	  move(const glm::vec3& vec);
	void	  scale(const glm::vec3& factor);
    void rotate_around(const float degree, const glm::vec3& axis, const glm::vec3& distance_to_object);
	glm::mat4 get_mat() const {
		return obj.get_mat();
	}
	void reset();
	void set_shader(const raw::shared_ptr<raw::shader>& sh);
	/**
	 * \brief
	 * draw the object
	 * \param reset should matrix reset? defaults to true
	 */
	void draw(decltype(drawing_method::drawing_method) drawing_method = drawing_method::basic,
			  bool									   reset		  = true);
};

} // namespace raw
#endif // SPACE_EXPLORER_SPHERE_H
