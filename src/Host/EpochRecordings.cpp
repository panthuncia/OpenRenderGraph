#include "OpenRenderGraph/EpochRecordings.h"

#include <atomic>
#include <exception>
#include <format>
#include <utility>

namespace org::epoch_recordings {

	namespace {
		std::atomic<std::uint64_t> requested{0};
		std::atomic<std::uint64_t> ready{0};
		std::atomic<std::uint64_t> dropped{0};
	}

	Result ResultOf(std::shared_ptr<const GraphHost::EpochRecording> a_recording, std::exception_ptr a_error, std::uint64_t a_buildGeneration) {
		Result result;
		if (a_error) {
			try {
				std::rethrow_exception(a_error);
			} catch (const GraphHost::EpochRecordingDropped& e) {
				result.outcome = Outcome::Dropped;
				result.error = e.what();
			} catch (const std::exception& e) {
				result.outcome = Outcome::Failed;
				result.error = e.what();
			} catch (...) {
				result.outcome = Outcome::Failed;
				result.error = "an unknown error";
			}
		} else if (!a_recording) {
			result.outcome = Outcome::Failed;
			result.error = "the host's thread completed the request with neither a recording nor an error";
		} else if (a_buildGeneration != 0 && a_recording->buildGeneration != a_buildGeneration) {
			result.outcome = Outcome::Dropped;
			result.error = std::format("recorded on graph build {}, not build {}", a_recording->buildGeneration, a_buildGeneration);
		} else {
			result.outcome = Outcome::Ready;
			result.recording = std::move(a_recording);
		}
		return result;
	}

	Posting Request(GraphHost& a_host, std::uint32_t a_epoch, std::shared_ptr<const IHostExecutionData> a_hostData,
		std::uint64_t a_buildGeneration, std::uint32_t a_copy, Done a_done) {
		const bool posted = a_host.RequestEpochRecording(
			a_epoch, std::move(a_hostData),
			[done = std::move(a_done), a_buildGeneration](std::shared_ptr<const GraphHost::EpochRecording> a_recording, std::exception_ptr a_error) {
				Result result = ResultOf(std::move(a_recording), std::move(a_error), a_buildGeneration);
				// A failure is not counted. It is the caller's to report.
				if (result.outcome == Outcome::Ready)
					ready.fetch_add(1, std::memory_order_relaxed);
				else if (result.outcome == Outcome::Dropped)
					dropped.fetch_add(1, std::memory_order_relaxed);
				done(std::move(result));
			},
			a_copy, a_buildGeneration);
		if (!posted)
			return Posting::Refused;
		requested.fetch_add(1, std::memory_order_relaxed);
		return Posting::Posted;
	}

	Stats Read() {
		return {requested.load(std::memory_order_relaxed), ready.load(std::memory_order_relaxed), dropped.load(std::memory_order_relaxed)};
	}
}
