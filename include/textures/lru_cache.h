//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_LRU_CACHE_H
#define SPACE_EXPLORER_LRU_CACHE_H

#include <raw_memory.h>

#include <list>
#include <unordered_map>

#include "textures/fwd.h"

namespace raw::texture {
template<typename K, typename V>
class lru_cache {
private:
	// Maximum amount of items that can be stored
	size_t capacity;

	std::list<std::pair<K, V> > list;
	std::unordered_map<K, typename decltype(list)::iterator> map;

public:
	explicit lru_cache(uint32_t cap) : capacity(cap) {
	}

	raw::shared_ptr<V> get(const K &key) {
		auto it = map.find(key);

		if (it == map.end()) {
			// Woopsie! cache miss
			return nullptr;
		}

		// Put the found element into the hot (first) cache (means it was just requested)
		list.splice(list.begin(), list, it->second);
		return raw::make_shared<V>(it->second->second);
	}

	void put(const K &key, V value) {
		auto it = map.find(key);
		if (it != map.end()) {
			// Update the value and put it to the hottest cache
			it->second->second = value;
			list.splice(list.begin(), list, it->second);
			return;
		}

		if (list.size() == capacity) {
			// Reached the cache capacity, delete last used element
			auto &last_node = list.back();
			map.erase(last_node.first);
			list.pop_back();
		}
		// Put new element
		list.emplace_front(key, value);
		map[key] = list.begin();
	}
};
} // namespace raw::texture

#endif // SPACE_EXPLORER_LRU_CACHE_H