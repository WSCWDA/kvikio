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

}  // namespace kvikio::detail
