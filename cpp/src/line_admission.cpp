/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <kvikio/line_admission.hpp>

namespace kvikio::detail {

class LineAdmission::Impl {
 public:
  Impl(std::size_t line_size, std::size_t sketch_bytes, std::size_t aging_interval,
       std::size_t minimum_accesses, std::uint64_t hit_ns, std::uint64_t fill_ns)
    : line_size{line_size}, aging_interval{aging_interval}, minimum_accesses{minimum_accesses},
      hit_ns{hit_ns}, fill_ns{fill_ns}, blocks(sketch_bytes / 64)
  {
    if (line_size == 0 || sketch_bytes < 64 || sketch_bytes % 64 != 0 ||
        aging_interval == 0 || minimum_accesses < 1 || minimum_accesses > 15) {
      throw std::invalid_argument("invalid line admission configuration");
    }
  }

  struct alignas(64) CounterBlock {
    std::array<std::uint8_t, 64> bytes{};
  };
  static_assert(sizeof(CounterBlock) == 64 && alignof(CounterBlock) == 64);

  static std::uint64_t mix(std::uint64_t x) noexcept
  {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  std::uint8_t get(std::size_t block, std::size_t slot) const noexcept
  {
    auto const byte = blocks[block].bytes[slot / 2];
    return (slot & 1) ? byte >> 4 : byte & 0x0f;
  }

  void set(std::size_t block, std::size_t slot, std::uint8_t value) noexcept
  {
    auto& byte = blocks[block].bytes[slot / 2];
    byte = (slot & 1) ? (byte & 0x0f) | (value << 4) : (byte & 0xf0) | value;
  }

  std::uint8_t record(std::size_t file_offset) noexcept
  {
    auto const key = file_offset / line_size;
    auto const block = mix(key) % blocks.size();
    std::array<std::size_t, 4> slots{};
    std::uint8_t estimate{15};
    for (std::size_t i = 0; i < slots.size(); ++i) {
      slots[i] = mix(key ^ (0x9e3779b97f4a7c15ULL * (i + 1))) % 128;
      estimate = std::min(estimate, get(block, slots[i]));
    }
    // Conservative update; repeated slot indices are incremented only once.
    if (estimate < 15) {
      for (auto const slot : slots) {
        if (get(block, slot) == estimate) { set(block, slot, estimate + 1); }
      }
    }
    // One 64-byte aging step per interval avoids a full-sketch pause.
    if (++requests % aging_interval == 0) {
      auto& data = blocks[aging_cursor].bytes;
      for (auto& byte : data) { byte = (byte & 0xee) >> 1; }
      aging_cursor = (aging_cursor + 1) % blocks.size();
      ++aging_count;
    }
    return estimate;
  }

  std::size_t line_size;
  std::size_t aging_interval;
  std::size_t minimum_accesses;
  std::uint64_t hit_ns;
  std::uint64_t fill_ns;
  std::vector<CounterBlock> blocks;
  mutable std::mutex mutex;
  std::uint64_t requests{};
  std::size_t aging_cursor{};
  std::uint64_t aging_count{};
  std::uint64_t bypass_count{};
  std::uint64_t benefit_bypass_count{};
  std::uint64_t admission_count{};
};

LineAdmission::LineAdmission(std::size_t line_size, std::size_t sketch_bytes,
                             std::size_t aging_interval, std::size_t minimum_accesses,
                             std::uint64_t hit_ns, std::uint64_t fill_ns)
  : _impl{std::make_unique<Impl>(line_size, sketch_bytes, aging_interval, minimum_accesses,
                                  hit_ns, fill_ns)}
{
}

LineAdmission::~LineAdmission() noexcept = default;

bool LineAdmission::should_admit(std::size_t file_offset, std::uint64_t bypass_ns)
{
  std::lock_guard lock{_impl->mutex};
  auto const estimate = _impl->record(file_offset);

  // Previous observations approximate the number of future uses. Never fill on a first touch.
  auto const saving = bypass_ns > _impl->hit_ns ? bypass_ns - _impl->hit_ns : 0;
  auto const extra_fill = _impl->fill_ns > bypass_ns ? _impl->fill_ns - bypass_ns : 0;
  bool const admit = _impl->minimum_accesses == 1 ||
                     (static_cast<std::size_t>(estimate) + 1 >= _impl->minimum_accesses &&
                      estimate > 0 && saving > 0 &&
                      static_cast<std::uint64_t>(estimate) * saving >= extra_fill);
  if (admit) { ++_impl->admission_count; }
  else {
    ++_impl->bypass_count;
    if (static_cast<std::size_t>(estimate) + 1 >= _impl->minimum_accesses) {
      ++_impl->benefit_bypass_count;
    }
  }

  return admit;
}

void LineAdmission::observe_hit(std::size_t file_offset)
{
  std::lock_guard lock{_impl->mutex};
  (void)_impl->record(file_offset);
}

void LineAdmission::clear() noexcept
{
  std::lock_guard lock{_impl->mutex};
  std::fill(_impl->blocks.begin(), _impl->blocks.end(), Impl::CounterBlock{});
  _impl->requests = 0;
  _impl->aging_cursor = 0;
}

std::uint64_t LineAdmission::bypasses() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  return _impl->bypass_count;
}

std::uint64_t LineAdmission::admissions() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  return _impl->admission_count;
}

std::uint64_t LineAdmission::benefit_bypasses() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  return _impl->benefit_bypass_count;
}

std::uint64_t LineAdmission::aging_steps() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  return _impl->aging_count;
}

namespace {

class WindowedBlockedSketch {
 public:
  struct alignas(64) CounterBlock {
    std::array<std::uint8_t, 64> bytes{};
  };

  WindowedBlockedSketch(std::size_t bytes, std::size_t window_requests, std::uint64_t salt)
    : _window_requests{window_requests}, _salt{salt}, _blocks(bytes / sizeof(CounterBlock))
  {
    if (bytes < sizeof(CounterBlock) || bytes % sizeof(CounterBlock) != 0 ||
        window_requests == 0) {
      throw std::invalid_argument("invalid frequency-momentum sketch configuration");
    }
  }

  static std::uint64_t mix(std::uint64_t x) noexcept
  {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  std::uint8_t record(std::uint64_t key) noexcept
  {
    // Spread one complete halving sweep uniformly over the configured window.
    _aging_budget += _blocks.size();
    while (_aging_budget >= _window_requests) {
      auto& data = _blocks[_aging_cursor].bytes;
      for (auto& byte : data) { byte = (byte & 0xee) >> 1; }
      _aging_cursor = (_aging_cursor + 1) % _blocks.size();
      _aging_budget -= _window_requests;
      ++_aging_steps;
    }

    auto const block = mix(key ^ _salt) % _blocks.size();
    std::array<std::size_t, 4> slots{};
    auto count = estimate(key);
    for (std::size_t i = 0; i < slots.size(); ++i) {
      slots[i] = mix(key ^ _salt ^ (0x9e3779b97f4a7c15ULL * (i + 1))) % 128;
    }
    if (count < 15) {
      for (auto const slot : slots) {
        if (get(block, slot) == count) { set(block, slot, count + 1); }
      }
      ++count;
    }
    return count;
  }

  [[nodiscard]] std::uint8_t estimate(std::uint64_t key) const noexcept
  {
    auto const block = mix(key ^ _salt) % _blocks.size();
    std::uint8_t result{15};
    for (std::size_t i = 0; i < 4; ++i) {
      auto const slot =
        mix(key ^ _salt ^ (0x9e3779b97f4a7c15ULL * (i + 1))) % 128;
      result = std::min(result, get(block, slot));
    }
    return result;
  }

  void clear() noexcept
  {
    std::fill(_blocks.begin(), _blocks.end(), CounterBlock{});
    _aging_budget = 0;
    _aging_cursor = 0;
  }

  [[nodiscard]] std::uint64_t aging_steps() const noexcept { return _aging_steps; }
  [[nodiscard]] std::size_t bytes() const noexcept
  {
    return _blocks.size() * sizeof(CounterBlock);
  }

 private:
  std::uint8_t get(std::size_t block, std::size_t slot) const noexcept
  {
    auto const byte = _blocks[block].bytes[slot / 2];
    return (slot & 1) ? byte >> 4 : byte & 0x0f;
  }

  void set(std::size_t block, std::size_t slot, std::uint8_t value) noexcept
  {
    auto& byte = _blocks[block].bytes[slot / 2];
    byte = (slot & 1) ? (byte & 0x0f) | (value << 4) : (byte & 0xf0) | value;
  }

  std::size_t _window_requests;
  std::uint64_t _salt;
  std::vector<CounterBlock> _blocks;
  std::size_t _aging_budget{};
  std::size_t _aging_cursor{};
  std::uint64_t _aging_steps{};
};

static_assert(sizeof(WindowedBlockedSketch::CounterBlock) == 64);
static_assert(alignof(WindowedBlockedSketch::CounterBlock) == 64);

}  // namespace

class FrequencyMomentumAdmission::Impl {
 public:
  Impl(std::size_t line_size,
       std::size_t frequency_bytes,
       std::size_t frequency_window_requests,
       std::size_t frequency_threshold,
       std::size_t momentum_bytes,
       std::size_t momentum_window_requests,
       std::size_t momentum_threshold)
    : line_size{line_size},
      frequency_threshold{frequency_threshold},
      momentum_threshold{momentum_threshold},
      frequency{frequency_bytes, frequency_window_requests, 0x243f6a8885a308d3ULL},
      momentum{momentum_bytes, momentum_window_requests, 0x13198a2e03707344ULL}
  {
    if (line_size == 0 || frequency_threshold == 0 || frequency_threshold > 15 ||
        momentum_threshold == 0 || momentum_threshold > 15) {
      throw std::invalid_argument("invalid frequency-momentum admission configuration");
    }
  }

  std::size_t line_size;
  std::size_t frequency_threshold;
  std::size_t momentum_threshold;
  WindowedBlockedSketch frequency;
  WindowedBlockedSketch momentum;
  mutable std::mutex mutex;
};

FrequencyMomentumAdmission::FrequencyMomentumAdmission(
  std::size_t line_size,
  std::size_t frequency_bytes,
  std::size_t frequency_window_requests,
  std::size_t frequency_threshold,
  std::size_t momentum_bytes,
  std::size_t momentum_window_requests,
  std::size_t momentum_threshold)
  : _impl{std::make_unique<Impl>(line_size,
                                 frequency_bytes,
                                 frequency_window_requests,
                                 frequency_threshold,
                                 momentum_bytes,
                                 momentum_window_requests,
                                 momentum_threshold)}
{
}

FrequencyMomentumAdmission::~FrequencyMomentumAdmission() noexcept = default;

FrequencyMomentumDecision FrequencyMomentumAdmission::observe(std::size_t file_offset)
{
  std::lock_guard lock{_impl->mutex};
  auto const key = file_offset / _impl->line_size;
  auto const frequency = _impl->frequency.record(key);
  auto const momentum  = _impl->momentum.record(key);
  bool const by_frequency = frequency >= _impl->frequency_threshold;
  bool const by_momentum  = momentum >= _impl->momentum_threshold;
  auto const signal = by_frequency && by_momentum ? AdmissionSignal::both
                      : by_frequency              ? AdmissionSignal::frequency
                      : by_momentum               ? AdmissionSignal::momentum
                                                  : AdmissionSignal::none;
  return {by_frequency || by_momentum, frequency, momentum, signal};
}

FrequencyMomentumDecision FrequencyMomentumAdmission::estimate(std::size_t file_offset) const
{
  std::lock_guard lock{_impl->mutex};
  auto const key       = file_offset / _impl->line_size;
  auto const frequency = _impl->frequency.estimate(key);
  auto const momentum  = _impl->momentum.estimate(key);
  bool const by_frequency = frequency >= _impl->frequency_threshold;
  bool const by_momentum  = momentum >= _impl->momentum_threshold;
  auto const signal = by_frequency && by_momentum ? AdmissionSignal::both
                      : by_frequency              ? AdmissionSignal::frequency
                      : by_momentum               ? AdmissionSignal::momentum
                                                  : AdmissionSignal::none;
  return {by_frequency || by_momentum, frequency, momentum, signal};
}

std::uint8_t FrequencyMomentumAdmission::observe_frequency(std::size_t file_offset)
{
  std::lock_guard lock{_impl->mutex};
  return _impl->frequency.record(file_offset / _impl->line_size);
}

std::uint8_t FrequencyMomentumAdmission::observe_momentum(std::size_t file_offset)
{
  std::lock_guard lock{_impl->mutex};
  return _impl->momentum.record(file_offset / _impl->line_size);
}

void FrequencyMomentumAdmission::clear() noexcept
{
  std::lock_guard lock{_impl->mutex};
  _impl->frequency.clear();
  _impl->momentum.clear();
}

std::uint64_t FrequencyMomentumAdmission::frequency_aging_steps() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  return _impl->frequency.aging_steps();
}

std::uint64_t FrequencyMomentumAdmission::momentum_aging_steps() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  return _impl->momentum.aging_steps();
}

std::size_t FrequencyMomentumAdmission::metadata_bytes() const noexcept
{
  std::lock_guard lock{_impl->mutex};
  return _impl->frequency.bytes() + _impl->momentum.bytes();
}

}  // namespace kvikio::detail
