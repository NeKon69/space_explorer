//
// Created by progamers on 8/28/25.
//

#ifndef SPACE_EXPLORER_GENERATION_CONTEST_H
#define SPACE_EXPLORER_GENERATION_CONTEST_H
namespace raw::common {
template<typename T, typename... Args>
concept IsResourceManager = requires(T t, Args... args) {
	{ t.prepare(args...) } -> std::same_as<void>;
	{ t.cleanup() } -> std::same_as<void>;
};

template<typename TResourceManager>
class scoped_resource_handle {
private:
	TResourceManager* manager;
	friend TResourceManager;

public:
	template<typename... Ts>
	explicit scoped_resource_handle(TResourceManager* mgr, Ts&&... args)
		requires IsResourceManager<TResourceManager, Ts...>
		: manager(mgr) {
		manager->prepare(std::forward<Ts>(args)...);
	}
	~scoped_resource_handle() {
		if (manager)
			manager->cleanup();
	}
	scoped_resource_handle(const scoped_resource_handle& other)				   = delete;
	scoped_resource_handle(scoped_resource_handle&& other) noexcept			   = default;
	scoped_resource_handle& operator=(const scoped_resource_handle& other)	   = delete;
	scoped_resource_handle& operator=(scoped_resource_handle&& other) noexcept = default;
};
} // namespace raw::common
#endif // SPACE_EXPLORER_GENERATION_CONTEST_H
