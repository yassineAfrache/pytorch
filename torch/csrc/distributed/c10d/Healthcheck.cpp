#include <sys/socket.h>
#include <future>

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/Healthcheck.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {

Healthcheck::Healthcheck(
    bool abortOnError,
    std::chrono::milliseconds interval,
    std::chrono::milliseconds timeout)
    : abortOnError_(abortOnError), interval_(interval), timeout_(timeout) {
  worker_ = std::async(std::launch::async, [this]() { runLoop(); });
}

void Healthcheck::runLoop() {
  for (int i = 0; i < 2; i++) {
    setup(i);
  }

  while (true) {
    C10D_DEBUG("Running healthchecks...");

    std::vector<std::future<void>> futures;
    futures.reserve(2);

    for (int i = 0; i < 2; i++) {
      futures.emplace_back(
          std::async(std::launch::async, [this, i]() { runHealthcheck(i); }));
    }

    // calculate deadline for the futures
    std::chrono::time_point<std::chrono::system_clock> deadline =
        std::chrono::system_clock::now() + interval_;

    int failures = 0;

    // wait for futures to complete
    for (auto& future : futures) {
      auto status = future.wait_until(deadline);
      if (status == std::future_status::timeout) {
        failures += 1;
        continue;
      }
      TORCH_INTERNAL_ASSERT(status == std::future_status::ready);

      try {
        future.get();
        C10D_DEBUG("Healthcheck passed");
      } catch (const std::exception& e) {
        C10D_ERROR("Healthcheck failed: {}", e.what());
        failures += 1;
        continue;
      } catch (...) {
        C10D_ERROR("Healthcheck failed with unknown exception");
        failures += 1;
        continue;
      }
    }

    if (failures == 2) {
      C10D_ERROR("Current host identified as problematic!");
      if (abortOnError_) {
        std::abort();
      }
    }

    // wait for interval
    {
      std::unique_lock lock{shutdownM_};
      shutdownCv_.wait_for(lock, interval_);
      if (shutdown_) {
        break;
      }
    }
  }
}

void Healthcheck::shutdown() {
  {
    std::unique_lock lock{shutdownM_};
    shutdown_ = true;
  }
  shutdownCv_.notify_all();

  worker_.get();
}

} // namespace c10d
