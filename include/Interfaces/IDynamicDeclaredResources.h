#pragma once

namespace org {

struct IDynamicDeclaredResources {
	virtual bool DeclaredResourcesChanged() const = 0;

	// Immediate-only passes use declarations for graph scheduling but do not resolve
	// resources through the pass view and have no declaration-dependent Setup work.
	// They can skip rebuilding those retained execution helpers after a refresh.
	virtual bool RequiresPassRebindAfterDeclarationRefresh() const noexcept { return true; }

	// The pass still uses DeclaredResourcesChanged() as its per-frame prepare hook,
	// but its immediate command list authoritatively declares every resource touched.
	virtual bool DeclarationsProvidedByImmediateCommands() const noexcept { return false; }

	// True (the default) when Prepare and Record use only what the bindings they are handed declare. For example, the
	// pass snapshots its work in Declare, puts that snapshot in its bindings, and Prepare takes work only from it; a
	// table it publishes goes through the layout and row count in its bindings, and it runs only the views those
	// declare.
	// While a new declaration builds, frames keep preparing against the old one. So the new declaration builds on a
	// worker, even when it adds resources. A pass that can't keep to this says false: when it adds resources, the graph
	// then builds inline, which stalls that frame.
	virtual bool PreparesOnlyDeclaredWork() const noexcept { return true; }

	// True when the resources the pass declares by pointer are places, not identities: a new declaration that names
	// other resources of the same shape in the same places (a texture recreated on resize, say) is the same declaration
	// with new backings. The graph then rebinds them, with no build, unless which slots share a backing changes.
	// Prepare and Record must be right for whatever resource each place has now: the new backings apply from the frame
	// the pass declares them. So such a pass must say PreparesOnlyDeclaredWork false: that one keeps old work on old
	// resources.
	virtual bool RebindsConcreteResources() const noexcept { return false; }
	virtual ~IDynamicDeclaredResources() = default;
};

} // namespace org
