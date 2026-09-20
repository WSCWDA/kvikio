/*
 * SPDX-License-Identifier: Apache-2.0
 * Standalone trace-replay ablation. LRU victim selection is identical across policies.
 */

#include <algorithm>
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
  int hot;
};
struct Entry {
  std::list<std::uint64_t>::iterator lru;
  std::size_t admission_id;
};
struct AdmissionRecord {
  std::uint64_t hits{};
  bool finalized{};
  bool low_value{};
  bool victim_reaccessed{};
};
std::uint64_t number(char const* value) { return std::stoull(value); }
double normalized_score(kvikio::detail::FrequencyMomentumDecision const& decision,
                        std::uint64_t frequency_threshold,
                        std::uint64_t momentum_threshold)
{
  return std::max(static_cast<double>(decision.frequency) / frequency_threshold,
                  static_cast<double>(decision.momentum) / momentum_threshold);
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 14) {
    std::cerr << "usage: replay TRACE POLICY CAPACITY FREQ_BYTES FREQ_WINDOW FREQ_THRESHOLD "
                 "MOM_BYTES MOM_WINDOW MOM_THRESHOLD LINE_SIZE HIT_NS FILL_NS BYPASS_NS\n";
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
  auto const hit_ns = number(argv[11]);
  auto const fill_ns = number(argv[12]);
  auto const bypass_ns = number(argv[13]);
  bool const valid_policy = policy == "cache_all" || policy == "frequency" ||
                            policy == "momentum" || policy == "hybrid" ||
                            policy == "dual_score";
  if (capacity == 0 || !valid_policy) { throw std::invalid_argument("invalid policy or capacity"); }

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
  std::vector<AdmissionRecord> admission_records;
  std::unordered_map<std::uint64_t, std::size_t> pending_evictions;
  using Episode = std::pair<std::size_t, std::uint64_t>;
  std::map<Episode, std::size_t> first_hot;
  std::map<Episode, std::size_t> detected_hot;

  auto const saved_per_hit = bypass_ns > hit_ns ? bypass_ns - hit_ns : 0;
  auto const extra_fill = fill_ns > bypass_ns ? fill_ns - bypass_ns : 0;
  std::uint64_t hits{}, misses{}, admissions{}, rejected{}, evictions{};
  std::uint64_t false_admissions{}, useful_admissions{}, low_value_admissions{};
  std::uint64_t pollution_misses{}, score_rejections{};
  std::uint64_t hot_hits{}, hot_requests{}, cold_requests{}, cold_admissions{}, hot_evictions{};
  std::uint64_t by_frequency{}, by_momentum{}, by_both{};
  auto finalize = [&](std::size_t admission_id) {
    auto& record = admission_records.at(admission_id);
    if (record.finalized) { return; }
    record.finalized = true;
    if (record.hits == 0) { ++false_admissions; }
    else { ++useful_admissions; }
    record.low_value = record.hits * saved_per_hit <= extra_fill;
    if (record.low_value) {
      ++low_value_admissions;
      if (record.victim_reaccessed) { ++pollution_misses; }
    }
  };

  auto const start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < events.size(); ++i) {
    auto const& current = events[i];
    kvikio::detail::FrequencyMomentumDecision decision{};
    if (policy == "frequency") {
      decision.frequency = tracker.observe_frequency(current.line * line_size);
    } else if (policy == "momentum") {
      decision.momentum = tracker.observe_momentum(current.line * line_size);
    } else if (policy == "hybrid" || policy == "dual_score") {
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
      ++admission_records[found->second.admission_id].hits;
      if (hot) {
        ++hot_hits;
        detected_hot.try_emplace(Episode{current.phase, current.line}, i);
      }
      lru.splice(lru.begin(), lru, found->second.lru);
      found->second.lru = lru.begin();
      continue;
    }
    ++misses;

    auto displaced = pending_evictions.find(current.line);
    if (displaced != pending_evictions.end()) {
      auto& cause = admission_records.at(displaced->second);
      cause.victim_reaccessed = true;
      if (cause.finalized && cause.low_value) { ++pollution_misses; }
      pending_evictions.erase(displaced);
    }

    bool admit = policy == "cache_all";
    if (policy == "frequency") { admit = decision.frequency >= frequency_threshold; }
    if (policy == "momentum") { admit = decision.momentum >= momentum_threshold; }
    if (policy == "hybrid" || policy == "dual_score") { admit = decision.admit; }
    if (admit && policy == "dual_score" && cache.size() == capacity) {
      auto const victim = lru.back();
      auto const victim_decision = tracker.estimate(victim * line_size);
      auto const candidate_score =
        normalized_score(decision, frequency_threshold, momentum_threshold);
      auto const victim_score =
        normalized_score(victim_decision, frequency_threshold, momentum_threshold);
      if (candidate_score <= victim_score) {
        admit = false;
        ++score_rejections;
      }
    }
    if (!admit) {
      ++rejected;
      continue;
    }

    ++admissions;
    if (!hot && current.hot >= 0) { ++cold_admissions; }
    if (policy == "frequency") { ++by_frequency; }
    else if (policy == "momentum") { ++by_momentum; }
    else if (policy != "cache_all") {
      if (decision.signal == kvikio::detail::AdmissionSignal::frequency) { ++by_frequency; }
      if (decision.signal == kvikio::detail::AdmissionSignal::momentum) { ++by_momentum; }
      if (decision.signal == kvikio::detail::AdmissionSignal::both) { ++by_both; }
    }
    if (hot) { detected_hot.try_emplace(Episode{current.phase, current.line}, i); }

    bool displaced_victim = false;
    std::uint64_t victim{};
    if (cache.size() == capacity) {
      victim = lru.back();
      auto const victim_entry = cache.find(victim);
      finalize(victim_entry->second.admission_id);
      if (labeled && hot_sets[current.phase].count(victim) != 0) { ++hot_evictions; }
      cache.erase(victim_entry);
      lru.pop_back();
      ++evictions;
      displaced_victim = true;
    }
    auto const admission_id = admission_records.size();
    admission_records.emplace_back();
    if (displaced_victim) { pending_evictions[victim] = admission_id; }
    lru.push_front(current.line);
    cache.emplace(current.line, Entry{lru.begin(), admission_id});
  }
  auto const finish = std::chrono::steady_clock::now();
  for (auto const& item : cache) { finalize(item.second.admission_id); }

  double detection_sum = 0;
  std::uint64_t detection_count = 0;
  if (labeled) {
    for (auto const& phase_lines : hot_sets) {
      for (auto const line : phase_lines.second) {
        Episode const key{phase_lines.first, line};
        auto const first = first_hot.at(key);
        auto const detected = detected_hot.find(key);
        auto const end = phase_end.at(phase_lines.first) + 1;
        detection_sum += static_cast<double>(
          (detected == detected_hot.end() ? end : detected->second) - first);
        ++detection_count;
      }
    }
  }

  auto const elapsed_ns = std::chrono::duration<double, std::nano>(finish - start).count();
  auto ratio = [](std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? 0.0 : static_cast<double>(numerator) / denominator;
  };
  auto const net_saved_ns = static_cast<std::int64_t>(hits * saved_per_hit) -
                            static_cast<std::int64_t>(admissions * extra_fill);
  std::cout << "{\"policy\":\"" << policy << "\",\"requests\":" << events.size()
            << ",\"hits\":" << hits << ",\"misses\":" << misses
            << ",\"hit_ratio\":" << ratio(hits, events.size())
            << ",\"admissions\":" << admissions << ",\"rejected\":" << rejected
            << ",\"score_rejections\":" << score_rejections << ",\"evictions\":" << evictions
            << ",\"false_admissions\":" << false_admissions
            << ",\"false_admission_rate\":" << ratio(false_admissions, admissions)
            << ",\"low_value_admissions\":" << low_value_admissions
            << ",\"low_value_admission\":" << ratio(low_value_admissions, admissions)
            << ",\"low_value_admission_rate\":" << ratio(low_value_admissions, admissions)
            << ",\"pollution_misses\":" << pollution_misses
            << ",\"net_saved_ns\":" << net_saved_ns
            << ",\"net_saved_ns_per_request\":" << static_cast<double>(net_saved_ns) / events.size()
            << ",\"useful_admissions\":" << useful_admissions
            << ",\"hot_hit_ratio\":"
            << (labeled ? std::to_string(ratio(hot_hits, hot_requests)) : "null")
            << ",\"cold_admission_rate\":"
            << (labeled ? std::to_string(ratio(cold_admissions, cold_requests)) : "null")
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
            << ",\"cost_model_ns\":{\"hit\":" << hit_ns << ",\"fill\":" << fill_ns
            << ",\"bypass\":" << bypass_ns << "}"
            << ",\"admission_reason\":{\"frequency\":" << by_frequency
            << ",\"momentum\":" << by_momentum << ",\"both\":" << by_both << "}}\n";
}
