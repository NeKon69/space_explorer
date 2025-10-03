//
// Created by progamers on 9/19/25.
//

#pragma once
#include "components/component.h"
#include "device_types/device_ptr.h"
#include "n_body/fwd.h"

namespace raw::n_body {
struct physics_component : components::component {};

template<typename T>
struct space_object_data {
	glm::vec<3, T> position;
	glm::vec<3, T> velocity;
	T			   mass;
	T			   radius;

	constexpr space_object_data()
		: position(0.0),
		  velocity(predef::BASIC_VELOCITY),
		  mass(predef::PLANET_MASS),
		  radius(predef::RADIUS) {}

	explicit constexpr space_object_data(glm::dvec3 _position,
										 glm::dvec3 _velocity = predef::BASIC_VELOCITY,
										 double		_mass	  = predef::PLANET_MASS,
										 double		_radius	  = predef::RADIUS)
		: position(_position), velocity(_velocity), mass(_mass), radius(_radius) {}

	space_object_data(space_object_data&& other)				 = default;
	space_object_data& operator=(space_object_data&& other)		 = default;
	space_object_data(const space_object_data& other)			 = default;
	space_object_data& operator=(const space_object_data& other) = default;
};

template<typename T>
struct soa_device_data {
	template<typename Tp>
	using device_pointer = device_types::device_ptr<Tp>;
	device_pointer<glm::vec<3, T>*> positions;
	device_pointer<glm::vec<3, T>*> velocities;
	device_pointer<T*>				masses;
	device_pointer<T*>				radii;
	soa_device_data() = delete;
	soa_device_data(device_pointer<glm::vec<3, T>*> _positions,
					device_pointer<glm::vec<3, T>*> _velocities, device_pointer<T*> _masses,
					device_pointer<T*> _radii)
		: positions(_positions), velocities(_velocities), masses(_masses), radii(_radii) {}
	soa_device_data(soa_device_data&&)						 = default;
	soa_device_data& operator=(soa_device_data&&)			 = default;
	soa_device_data(const soa_device_data& other)			 = default;
	soa_device_data& operator=(const soa_device_data& other) = default;
};

} // namespace raw::n_body