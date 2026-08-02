#pragma once

struct IDynamicDeclaredResources {
	virtual bool DeclaredResourcesChanged() const = 0;

	// Immediate-only passes use declarations for graph scheduling but do not resolve
	// resources through the pass view and have no declaration-dependent Setup work.
	// They can skip rebuilding those retained execution helpers after a refresh.
	virtual bool RequiresPassRebindAfterDeclarationRefresh() const noexcept { return true; }

	// The pass still uses DeclaredResourcesChanged() as its per-frame prepare hook,
	// but its immediate command list authoritatively declares every resource touched.
	virtual bool DeclarationsProvidedByImmediateCommands() const noexcept { return false; }
	virtual ~IDynamicDeclaredResources() = default;
};
