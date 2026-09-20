/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace kvikio::detail {

/** Fixed-size, cache-line-keyed admission tracker. No metadata is allocated per file line. */
class LineAdmission {
 public:
  LineAdmission(std::size_t line_size, std::size_t sketch_bytes, std::size_t aging_interval,
                std::size_t minimum_accesses, std::uint64_t hit_ns, std::uint64_t fill_ns);
  ~LineAdmission() noexcept;
  LineAdmission(LineAdmission const&) = delete;
  LineAdmission& operator=(LineAdmission const&) = delete;

  // bypass_ns is the cost of the actual fallback path for this logical request.
  [[nodiscard]] bool should_admit(std::size_t file_offset, std::uint64_t bypass_ns);
  void observe_hit(std::size_t file_offset);
  void clear() noexcept;
  [[nodiscard]] std::uint64_t bypasses() const noexcept;
  [[nodiscard]] std::uint64_t benefit_bypasses() const noexcept;
  [[nodiscard]] std::uint64_t admissions() const noexcept;
  [[nodiscard]] std::uint64_t aging_steps() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

enum class AdmissionSignal : std::uint8_t {
  none      = 0,
  frequency = 1,
  momentum  = 2,
  both      = 3,
};

struct FrequencyMomentumDecision {
  bool admit{};
  std::uint8_t frequency{};
  std::uint8_t momentum{};
  AdmissionSignal signal{AdmissionSignal::none};
};

/**
 * Two-timescale blocked-CBF tracker for admission-policy ablations.
 *
 * `*_window_requests` is the number of observations required to age every
 * 64-byte block once. Consequently, changing sketch capacity changes the
 * collision rate without silently changing the configured history window.
 */
class FrequencyMomentumAdmission {
 public:
  FrequencyMomentumAdmission(std::size_t line_size,
                             std::size_t frequency_bytes,
                             std::size_t frequency_window_requests,
                             std::size_t frequency_threshold,
                             std::size_t momentum_bytes,
                             std::size_t momentum_window_requests,
                             std::size_t momentum_threshold);
  ~FrequencyMomentumAdmission() noexcept;
  FrequencyMomentumAdmission(FrequencyMomentumAdmission const&) = delete;
  FrequencyMomentumAdmission& operator=(FrequencyMomentumAdmission const&) = delete;

  [[nodiscard]] FrequencyMomentumDecision observe(std::size_t file_offset);
  [[nodiscard]] std::uint8_t observe_frequency(std::size_t file_offset);
  [[nodiscard]] std::uint8_t observe_momentum(std::size_t file_offset);
  void clear() noexcept;
  [[nodiscard]] std::uint64_t frequency_aging_steps() const noexcept;
  [[nodiscard]] std::uint64_t momentum_aging_steps() const noexcept;
  [[nodiscard]] std::size_t metadata_bytes() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace kvikio::detail
