/*
 * SPDX-License-Identifier: Apache-2.0
 * Standalone trace-replay ablation for cache admission. It intentionally keeps
 * LRU identical across policies and performs no storage or CUDA I/O.
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <kvikio/line_admission.hpp>

namespace {

struct Event {
  std::uint64_t line;
  std::size_t phase;
  int hot;  // 0/1 for synthetic labeled traces; -1 for an unlabeled real trace.
};

struct Entry {
  std::list<std::uint64_t>::iterator lru;
  std::uint64_t hits{};
};

std::uint64_t number(char const* value) { return std::stoull(value); }

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 11) {
    std::cerr << "usage: replay TRACE POLICY CAPACITY FREQ_BYTES FREQ_WINDOW FREQ_THRESHOLD "
                 "MOM_BYTES MOM_WINDOW MOM_THRESHOLD LINE_SIZE\n";
    return 2;
  }
  auto const trace_path = argv[1];
  std::string const policy{argv[2]};
  auto const capacity = number(argv[3]);
  auto const frequency_bytes = number(argv[4]);
  auto const frequency_window = number(argv[5]);
  auto const frequency_threshold = number(argv[6]);
  auto const momentum_bytes = number(argv[7]);
  auto const momentum_window = number(argv[8]);
  auto const momentum_threshold = number(argv[9]);
  auto const line_size = number(argv[10]);
  if (capacity == 0 || (policy != "cache_all" && policy != "frequency" &&
                        policy != "momentum" && policy != "hybrid")) {
    throw std::invalid_argument("invalid policy or capacity");
  }

  std::ifstream input{trace_path};
  if (!input) { throw std::runtime_error("cannot open trace"); }
  std::vector<Event> events;
  Event event{};
  while (input >> event.line >> event.phase >> event.hot) { events.push_back(event); }
  if (events.empty() || !input.eof()) { throw std::runtime_error("empty or malformed trace"); }

  std::unordered_map<std::size_t, std::unordered_set<std::uint64_t>> hot_sets;
  std::unordered_map<std::size_t, std::size_t> phase_end;
  bool labeled = false;
  for (std::size_t i = 0; i < events.size(); ++i) {
    phase_end[events[i].phase] = i;
    if (events[i].hot >= 0) {
      labeled = true;
      if (events[i].hot != 0) { hot_sets[events[i].phase].insert(events[i].line); }
    }
  }

  kvikio::detail::FrequencyMomentumAdmission tracker{
    line_size, frequency_bytes, frequency_window, frequency_threshold,
    momentum_bytes, momentum_window, momentum_threshold};
  std::list<std::uint64_t> lru;
  std::unordered_map<std::uint64_t, Entry> cache;
  using Episode = std::pair<std::size_t, std::uint64_t>;
  std::map<Episode, std::size_t> first_hot;
  std::map<Episode, std::size_t> detected_hot;

  std::uint64_t hits{}, misses{}, admissions{}, rejected{}, evictions{};
  std::uint64_t false_admissions{}, useful_admissions{}, hot_hits{}, hot_requests{};
  std::uint64_t cold_requests{}, cold_admissions{}, hot_evictions{};
  std::uint64_t by_frequency{}, by_momentum{}, by_both{};
  auto finalize = [&](Entry const& entry) {
    if (entry.hits == 0) { ++false_admissions; }
    else { ++useful_admissions; }
  };

  auto const start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < events.size(); ++i) {
    auto const& current = events[i];
    kvikio::detail::FrequencyMomentumDecision decision{};
    if (policy == "frequency") {
      decision.frequency = tracker.observe_frequency(current.line * line_size);
    } else if (policy == "momentum") {
      decision.momentum = tracker.observe_momentum(current.line * line_size);
    } else if (policy == "hybrid") {
      decision = tracker.observe(current.line * line_size);
    }
    auto const hot = current.hot > 0;
    if (current.hot >= 0) {
      hot ? ++hot_requests : ++cold_requests;
      if (hot) { first_hot.try_emplace(Episode{current.phase, current.line}, i); }
    }

    auto found = cache.find(current.line);
    if (found != cache.end()) {
      ++hits;
      ++found->second.hits;
      if (hot) { ++hot_hits; }
      if (hot) { detected_hot.try_emplace(Episode{current.phase, current.line}, i); }
      lru.splice(lru.begin(), lru, found->second.lru);
      found->second.lru = lru.begin();
      continue;
    }
    ++misses;

    bool admit = policy == "cache_all";
    if (policy == "frequency") { admit = decision.frequency >= frequency_threshold; }
    if (policy == "momentum") { admit = decision.momentum >= momentum_threshold; }
    if (policy == "hybrid") { admit = decision.admit; }
    if (!admit) {
      ++rejected;
      continue;
    }
    ++admissions;
    if (!hot && current.hot >= 0) { ++cold_admissions; }
    if (policy == "frequency") { ++by_frequency; }
    else if (policy == "momentum") { ++by_momentum; }
    else {
      if (decision.signal == kvikio::detail::AdmissionSignal::frequency) { ++by_frequency; }
      if (decision.signal == kvikio::detail::AdmissionSignal::momentum) { ++by_momentum; }
      if (decision.signal == kvikio::detail::AdmissionSignal::both) { ++by_both; }
    }
    if (hot) { detected_hot.try_emplace(Episode{current.phase, current.line}, i); }

    if (cache.size() == capacity) {
      auto const victim = lru.back();
      auto const victim_entry = cache.find(victim);
      finalize(victim_entry->second);
      if (labeled && hot_sets[current.phase].count(victim) != 0) { ++hot_evictions; }
      cache.erase(victim_entry);
      lru.pop_back();
      ++evictions;
    }
    lru.push_front(current.line);
    cache.emplace(current.line, Entry{lru.begin(), 0});
  }
  auto const finish = std::chrono::steady_clock::now();
  for (auto const& [line, entry] : cache) { finalize(entry); }

  double detection_sum = 0;
  std::uint64_t detection_count = 0;
  if (labeled) {
    for (auto const& [phase, lines] : hot_sets) {
      for (auto const line : lines) {
        Episode const key{phase, line};
        auto const first = first_hot.at(key);
        auto const detected = detected_hot.find(key);
        auto const end = phase_end.at(phase) + 1;
        detection_sum += static_cast<double>((detected == detected_hot.end() ? end : detected->second) - first);
        ++detection_count;
      }
    }
  }

  auto const elapsed_ns = std::chrono::duration<double, std::nano>(finish - start).count();
  auto ratio = [](std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? 0.0 : static_cast<double>(numerator) / denominator;
  };
  std::cout << "{\"policy\":\"" << policy << "\",\"requests\":" << events.size()
            << ",\"hits\":" << hits << ",\"misses\":" << misses
            << ",\"hit_ratio\":" << ratio(hits, events.size())
            << ",\"admissions\":" << admissions << ",\"rejected\":" << rejected
            << ",\"evictions\":" << evictions
            << ",\"false_admissions\":" << false_admissions
            << ",\"false_admission_rate\":" << ratio(false_admissions, admissions)
            << ",\"useful_admissions\":" << useful_admissions
            << ",\"hot_hit_ratio\":" << (labeled ? std::to_string(ratio(hot_hits, hot_requests)) : "null")
            << ",\"cold_admission_rate\":" << (labeled ? std::to_string(ratio(cold_admissions, cold_requests)) : "null")
            << ",\"hot_evictions\":" << hot_evictions
            << ",\"mean_detection_delay_requests\":"
            << (labeled ? std::to_string(detection_sum / detection_count) : "null")
            << ",\"hits_per_admission\":" << ratio(hits, admissions)
            << ",\"decision_ns_per_request\":" << elapsed_ns / events.size()
            << ",\"frequency_aging_steps\":" << tracker.frequency_aging_steps()
            << ",\"momentum_aging_steps\":" << tracker.momentum_aging_steps()
            << ",\"metadata_bytes\":"
            << (policy == "cache_all" ? 0 : policy == "frequency" ? frequency_bytes
                                       : policy == "momentum" ? momentum_bytes
                                                               : tracker.metadata_bytes())
            << ",\"admission_reason\":{\"frequency\":" << by_frequency
            << ",\"momentum\":" << by_momentum << ",\"both\":" << by_both << "}}\n";
}
