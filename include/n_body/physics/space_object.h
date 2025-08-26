//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_SPACE_OBJECT_H
#define SPACE_EXPLORER_SPACE_OBJECT_H
#include <glm/glm.hpp>

#include "helper/clock.h"
#include "cuda_types/fwd.h"
#include "launch_leapfrog.h"
#include "../n_body_predef.h"

namespace raw {
    namespace predef {
        PASSIVE_VALUE BASIC_VELOCITY = glm::vec3(0.0f, 0.0f, 0.0f);
        PASSIVE_VALUE BASIC_ACCELERATION = glm::vec3(0.0f);
        PASSIVE_VALUE PLANET_MASS = 1.0;
        PASSIVE_VALUE RADIUS = 1.0;
    } // namespace predef

    template<typename T>
    struct space_object_data {
        glm::vec<3, T> position;
        glm::vec<3, T> velocity;
        T mass;
        T radius;

        space_object_data()
            : position(0.0),
              velocity(predef::BASIC_VELOCITY),
              mass(predef::PLANET_MASS),
              radius(predef::RADIUS) {
        }

        explicit space_object_data(glm::dvec3 _position, glm::dvec3 _velocity = predef::BASIC_VELOCITY,
                                   double _mass = predef::PLANET_MASS, double _radius = predef::RADIUS)
            : position(_position), velocity(_velocity), mass(_mass), radius(_radius) {
        }
    };

    template<typename T>
    class space_object {
    private:
        friend struct space_object_data<T>;
        friend class interaction_system<T>;

    public:
        space_object_data<T> object_data;

	__device__ __host__ space_object_data<T> &get() {
            return object_data;
        }

        space_object(const space_object &object) = default;

        space_object &operator=(const space_object &object) = default;

        space_object() : object_data(glm::vec < 3, T > (1.0)) {
        }

        explicit space_object(glm::dvec3 _position, glm::dvec3 _velocity = predef::BASIC_VELOCITY,
                              double _mass = predef::PLANET_MASS, double _radius = predef::RADIUS)
            : object_data(_position, _velocity, _mass, _radius) {
        }

        static void update_position(space_object *data, glm::mat4 *data_model, time since_last_upd,
                                    unsigned int count, const shared_ptr<cuda_stream> &stream) {
            since_last_upd.to_milli();
            launch_leapfrog<T>(data, data_model, since_last_upd.val, count, predef::G,
                               stream->stream());
        };
    };
} // namespace raw

#endif // SPACE_EXPLORER_SPACE_OBJECT_H