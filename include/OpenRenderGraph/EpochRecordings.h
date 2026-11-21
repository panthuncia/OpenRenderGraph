#pragma once

// Requests for revision-driven epoch recordings (GraphHost::RequestEpochRecording) with one set of outcome rules, and a
// cache that keeps a caller's recordings by what they were recorded for.

#include "OpenRenderGraph/GraphHost.h"
#include "Render/RunRef.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace org::epoch_recordings {

	/** @brief How a request ended. */
	enum class Outcome : std::uint8_t {
		Ready,
		Dropped,
		Failed
	};

	/** @brief What a request ended with. */
	struct Result {
		Outcome outcome = Outcome::Failed;
		std::shared_ptr<const GraphHost::EpochRecording> recording;  // Ready only
		std::string error;                                           // Dropped or Failed: why
	};

	/** @brief Whether the host took a request. */
	enum class Posting : std::uint8_t {
		Posted,
		Refused  // the host takes no requests: async epochs were never started or are off
	};

	// Called once, on the host's thread, or inline on the posting thread.
	using Done = std::function<void(Result)>;

	/**
	 * @brief The outcome rules, applied to what GraphHost::RequestEpochRecording completed with. Any thread.
	 * - EpochRecordingDropped, or a recording whose buildGeneration is not a_buildGeneration (when a_buildGeneration is
	 *   not 0): Dropped. The caller asks again.
	 * - Any other error, or neither a recording nor an error: Failed.
	 * - Otherwise: Ready.
	 */
	Result ResultOf(std::shared_ptr<const GraphHost::EpochRecording> a_recording, std::exception_ptr a_error, std::uint64_t a_buildGeneration);

	/**
	 * @brief Any thread. Posts one recording of a_epoch for a_hostData on build a_buildGeneration (0: the attached build),
	 * copy a_copy (RunRef::kNoCopy: every copy). a_done gets the result by ResultOf's rules, Failed included.
	 * Refused when the host takes no requests. a_done is not called then.
	 */
	Posting Request(GraphHost& a_host, std::uint32_t a_epoch, std::shared_ptr<const IHostExecutionData> a_hostData,
		std::uint64_t a_buildGeneration, std::uint32_t a_copy, Done a_done);

	/** @brief The builds of an epoch whose recordings a cache keeps. 0 means none. */
	struct Builds {
		std::uint64_t attached = 0;
		std::uint64_t staged = 0;
		/** @brief Whether a recording on build a_build is of one of them. */
		bool Has(std::uint64_t a_build) const { return a_build != 0 && (a_build == attached || a_build == staged); }
	};

	/** @brief What a cache does with an entry that a graph build dropped. */
	enum class AfterDrop : std::uint8_t {
		AskAgain,        // the next lookup of its key misses, so the caller asks again
		KeepUntilNewKey  // the lookup finds it dropped. The caller asks again only for a new key
	};

	// Called when a recording failed. a_what names it ("the recording of ..."), a_error says why. After it returns, the
	// cache treats the failure as described at each call.
	using OnFailure = std::function<void(std::string_view a_what, std::string_view a_error)>;

	/**
	 * @brief One caller's kept epoch recordings, keyed by what they were recorded for.
	 *
	 * Key needs operator==. Hash is a default-constructible function object that returns a std::uint64_t for a Key.
	 * An entry is one or more requests (variants) that settle together. The owner thread finds, adds and erases entries.
	 * The host's thread settles them under the cache's lock. It reaches only the entry's settle state, never the key or
	 * the payload. The rules:
	 * - a request that fails goes to OnFailure when it settles. Its entry settles with the error;
	 * - a kept entry that settled with no recording and no drop goes to OnFailure when it is looked up. The lookup misses;
	 * - a dropped entry follows Rule;
	 * - adding an entry first drops the settled entries recorded on a build the caller no longer keeps (Builds). They hold
	 *   that build's graph and alias pools;
	 * - when the cache holds Keep entries, the least recently used settled one goes.
	 * Payload is the caller's own data per entry, such as what its recordings resolve.
	 */
	template <class Key, std::size_t Keep, AfterDrop Rule, class Hash, class Payload = std::monostate>
	class Cache {
	public:
		using Recording = std::shared_ptr<const GraphHost::EpochRecording>;

		/** @brief An entry's state. */
		struct Status {
			bool settled = false;               // every request is recorded, dropped, failed or refused
			bool dropped = false;               // a graph build dropped a request. This is not a failure
			std::vector<Recording> recordings;  // by request. Kept only when every request recorded
			std::string error;                  // why it was dropped, failed or refused
			/** @brief Whether every request recorded. */
			bool Ready() const { return settled && !dropped && error.empty(); }
		};

		class Entry;
		using EntryPtr = std::shared_ptr<Entry>;

		/** @brief One kept entry. The key and hash are fixed. lastUse and payload belong to the owner thread. */
		class Entry {
		public:
			Key key;
			std::uint64_t hash = 0;     // Hash of key
			std::uint64_t lastUse = 0;  // the caller's clock at the last lookup
			Payload payload{};          // the caller's own data, set after Add

			/** @brief The entry's state. Only under the cache's lock, inside AnyOf and EraseIf. */
			const Status& LockedStatus() const { return m_settle->status; }

		private:
			friend class Cache;
			// Under the cache's lock. The host's thread holds this part alone.
			struct Settle {
				Status status;
				std::size_t remaining = 0;
			};
			std::shared_ptr<Settle> m_settle = std::make_shared<Settle>();
		};

		// Called once per entry, on the thread that settled it, outside the lock.
		using Settled = std::function<void(const Status&)>;
		// Posts request a_index of an entry. Calls a_done once unless it returns Refused.
		using Requester = std::function<Posting(std::size_t a_index, Done a_done)>;

		/** @brief a_onFailure hears about failed recordings. Without one, failures act as described and are not reported. */
		explicit Cache(OnFailure a_onFailure = {}) :
			m_shared(std::make_shared<Shared>()) { m_shared->onFailure = std::move(a_onFailure); }

		/** @brief Hash's value for a_key. */
		static std::uint64_t HashOf(const Key& a_key) { return static_cast<std::uint64_t>(Hash{}(a_key)); }

		/**
		 * @brief Owner thread. The kept entry for a_key, or null.
		 * a_unusable(const Status&) drops a ready entry the caller cannot use any more, so the lookup misses.
		 * a_describe() names the entry for OnFailure when a kept entry failed. That entry goes and the lookup misses.
		 */
		template <class Unusable, class Describe>
		EntryPtr Find(const Key& a_key, std::uint64_t a_use, Unusable&& a_unusable, Describe&& a_describe) {
			const std::uint64_t hash = HashOf(a_key);
			std::string failure;
			EntryPtr found;
			{
				const std::lock_guard lock(m_shared->mutex);
				for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
					auto& kept = **it;
					if (kept.hash != hash || !(kept.key == a_key))
						continue;
					const Status& status = kept.m_settle->status;
					if (status.settled && !status.Ready()) {
						if (!status.dropped) {
							failure = status.error.empty() ? std::string("no recording") : status.error;
							m_entries.erase(it);
							break;
						}
						if constexpr (Rule == AfterDrop::AskAgain) {
							m_entries.erase(it);
							return nullptr;
						}
					} else if (status.Ready() && a_unusable(std::as_const(status))) {
						m_entries.erase(it);
						return nullptr;
					}
					kept.lastUse = a_use;
					found = *it;
					break;
				}
			}
			if (!failure.empty())
				ReportFailure(*m_shared, std::format("the recording of {}", a_describe()), failure);
			return found;
		}

		/**
		 * @brief Owner thread. A new entry for a_key with a_requests requests to post. It replaces nothing, so call Find first.
		 * First drops the settled entries recorded on builds not in a_builds, then the least recently used.
		 */
		EntryPtr Add(const Builds& a_builds, Key a_key, std::uint64_t a_use, std::size_t a_requests = 1) {
			auto entry = std::make_shared<Entry>();
			entry->hash = HashOf(a_key);
			entry->key = std::move(a_key);
			entry->lastUse = a_use;
			entry->m_settle->status.recordings.resize(a_requests);
			entry->m_settle->remaining = a_requests;
			const std::lock_guard lock(m_shared->mutex);
			std::erase_if(m_entries, [&](const EntryPtr& a_kept) {
				const Status& status = a_kept->m_settle->status;
				return status.Ready() && std::ranges::any_of(status.recordings, [&](const Recording& a_recording) {
					return a_recording && !a_builds.Has(a_recording->buildGeneration);
				});
			});
			while (m_entries.size() >= Keep) {
				auto oldest = m_entries.end();
				for (auto it = m_entries.begin(); it != m_entries.end(); ++it)
					if ((*it)->m_settle->status.settled && (oldest == m_entries.end() || (*it)->lastUse < (*oldest)->lastUse))
						oldest = it;
				if (oldest == m_entries.end())
					break;
				m_entries.erase(oldest);
			}
			m_entries.push_back(entry);
			return entry;
		}

		/**
		 * @brief Any thread. Posts a_entry's a_count requests through a_request, in order. a_what names them for OnFailure.
		 * With more than one request, each gets " (request i)". a_settled runs once, when the last one settles.
		 * A refused post settles the requests not posted with an error and returns Refused. The requests after it are not posted.
		 */
		Posting Post(const EntryPtr& a_entry, std::size_t a_count, std::string_view a_what, const Requester& a_request, Settled a_settled = {}) {
			auto settled = std::make_shared<const Settled>(std::move(a_settled));
			for (std::size_t i = 0; i < a_count; ++i) {
				auto what = a_count > 1 ? std::format("{} (request {})", a_what, i) : std::string(a_what);
				const Posting posted = a_request(i, [shared = m_shared, settle = a_entry->m_settle, settled, i, what = std::move(what)](Result a_result) {
					if (a_result.outcome == Outcome::Failed)
						ReportFailure(*shared, std::format("the recording of {}", what), a_result.error);
					Complete(shared->mutex, *settle, *settled, i, std::move(a_result), 1);
				});
				if (posted == Posting::Refused) {
					Result refused;
					refused.outcome = Outcome::Failed;
					refused.error = "the host takes no recording requests (async epochs are off)";
					Complete(m_shared->mutex, *a_entry->m_settle, *settled, i, std::move(refused), a_count - i);
					return Posting::Refused;
				}
			}
			return Posting::Posted;
		}

		/**
		 * @brief Any thread. Posts a_entry's requests to a_host through Request, one per host data, on build
		 * a_buildGeneration and copy a_copy. Otherwise as the Post above.
		 */
		Posting Post(GraphHost& a_host, const EntryPtr& a_entry, std::uint32_t a_epoch, std::vector<std::shared_ptr<const IHostExecutionData>> a_hostData,
			std::uint64_t a_buildGeneration, std::uint32_t a_copy, std::string_view a_what, Settled a_settled = {}) {
			return Post(
				a_entry, a_hostData.size(), a_what,
				[&](std::size_t a_index, Done a_done) {
					return Request(a_host, a_epoch, std::move(a_hostData[a_index]), a_buildGeneration, a_copy, std::move(a_done));
				},
				std::move(a_settled));
		}

		/** @brief Any thread. A copy of a_entry's state. */
		Status Read(const Entry& a_entry) const {
			const std::lock_guard lock(m_shared->mutex);
			return a_entry.m_settle->status;
		}

		/** @brief Any thread. Whether every request of a_entry settled. */
		bool IsSettled(const Entry& a_entry) const {
			const std::lock_guard lock(m_shared->mutex);
			return a_entry.m_settle->status.settled;
		}

		/** @brief Any thread. a_entry's recording a_index once every request recorded, else null. */
		Recording RecordingOf(const Entry& a_entry, std::size_t a_index = 0) const {
			const std::lock_guard lock(m_shared->mutex);
			const Status& status = a_entry.m_settle->status;
			return status.Ready() && a_index < status.recordings.size() ? status.recordings[a_index] : nullptr;
		}

		/** @brief Owner thread. Whether a kept entry matches a_pred(const Entry&). a_pred runs under the lock. */
		template <class Pred>
		bool AnyOf(Pred&& a_pred) const {
			const std::lock_guard lock(m_shared->mutex);
			return std::ranges::any_of(m_entries, [&](const EntryPtr& a_entry) { return a_pred(std::as_const(*a_entry)); });
		}

		/** @brief Owner thread. Erases the kept entries a_pred(const Entry&) matches. a_pred runs under the lock. Returns how many. */
		template <class Pred>
		std::size_t EraseIf(Pred&& a_pred) {
			const std::lock_guard lock(m_shared->mutex);
			return static_cast<std::size_t>(std::erase_if(m_entries, [&](const EntryPtr& a_entry) { return a_pred(std::as_const(*a_entry)); }));
		}

	private:
		// Shared with the requests in flight, so a request may settle after the cache is gone.
		struct Shared {
			std::mutex mutex;
			OnFailure onFailure;
		};

		static void ReportFailure(const Shared& a_shared, std::string_view a_what, std::string_view a_error) {
			if (a_shared.onFailure)
				a_shared.onFailure(a_what, a_error.empty() ? std::string_view("an unknown error") : a_error);
		}

		/** @brief a_count requests from a_index settled with a_result. The entry settles with the last one. */
		static void Complete(std::mutex& a_mutex, typename Entry::Settle& a_settle, const Settled& a_settled, std::size_t a_index, Result a_result,
			std::size_t a_count) {
			Status done;
			{
				const std::lock_guard lock(a_mutex);
				auto& status = a_settle.status;
				if (a_result.outcome == Outcome::Ready) {
					status.recordings[a_index] = std::move(a_result.recording);
				} else {
					status.dropped = status.dropped || a_result.outcome == Outcome::Dropped;
					if (status.error.empty())
						status.error = std::move(a_result.error);
				}
				a_settle.remaining -= (std::min)(a_count, a_settle.remaining);
				if (a_settle.remaining != 0)
					return;
				status.settled = true;
				// A set of recordings is used whole or not at all.
				if (!status.Ready())
					status.recordings.clear();
				if (!a_settled)
					return;
				done = status;
			}
			a_settled(done);
		}

		std::shared_ptr<Shared> m_shared;
		std::vector<EntryPtr> m_entries;
	};

	/** @brief Counters of every request so far, for a host's exit report. */
	struct Stats {
		std::uint64_t requested = 0;
		std::uint64_t ready = 0;
		std::uint64_t dropped = 0;
	};

	/** @brief Any thread. The counters so far, over every host in the process. */
	Stats Read();
}
