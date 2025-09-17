//
// Created by progamers on 8/5/25.
//

#pragma once

#include <raw_memory.h>

#include <functional>
#include <list>
#include <unordered_map>

#include "texture_generation/fwd.h"

namespace raw::texture_generation {
template<typename K, typename V>
class lru_cache {
private:
	// Maximum amount of items that can be stored
	size_t capacity;

	std::list<std::pair<K, V> >								 list;
	std::unordered_map<K, typename decltype(list)::iterator> map;
	std::function<void(const K &, const V &)>				 eviction_callback;

public:
	lru_cache() = default;
	explicit lru_cache(uint32_t cap, decltype(eviction_callback) on_evict = nullptr)
		: capacity(cap), eviction_callback(on_evict) {}

	std::shared_ptr<V> get(const K &key) {
		auto it = map.find(key);

		if (it == map.end()) {
			// Woopsie! cache miss
			return nullptr;
		}

		// Put the found element into the hot (first) cache (means it was just requested)
		list.splice(list.begin(), list, it->second);
		return std::make_shared<V>(it->second->second);
	}

	void put(const K &key, V value) {
		auto it = map.find(key);
		if (it != map.end()) {
			// Update the value and put it to the hottest cache
			it->second->second = std::move(value);
			list.splice(list.begin(), list, it->second);
			return;
		}

		if (list.size() == capacity) {
			// Reached the cache capacity, delete last used element
			auto &last_node = list.back();
			if (eviction_callback) {
				eviction_callback(last_node.first, last_node.second);
			}
			map.erase(last_node.first);
			list.pop_back();
		}
		// Put new element
		list.emplace_front(key, value);
		map[key] = list.begin();
	}
	lru_cache(const lru_cache &)			= delete;
	lru_cache &operator=(const lru_cache &) = delete;
	lru_cache(lru_cache &&)					= default;
	lru_cache &operator=(lru_cache &&)		= default;
};
} // namespace raw::texture_generation
