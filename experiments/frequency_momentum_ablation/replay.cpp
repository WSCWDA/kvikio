/*
 * SPDX-License-Identifier: Apache-2.0
 * Standalone trace-replay ablation. The production HostCache is not involved.
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <list>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <kvikio/line_admission.hpp>

namespace {
enum class RequestClass : int { unknown = -1, background = 0, hot = 1, scan = 2, burst = 3 };
enum class ScoreMode { none, maximum, weighted, multiplicative, lexicographic };

struct Event {
  std::uint64_t line{};
  std::size_t phase{};
  int hot{-1};
  RequestClass request_class{RequestClass::unknown};
};
struct Entry {
  std::list<std::uint64_t>::iterator lru;
  std::size_t admission_id{};
};
struct AdmissionRecord {
  std::uint64_t hits{};
  bool finalized{};
  bool low_value{};
  bool victim_reaccessed{};
};

class ShadowCache {
 public:
  explicit ShadowCache(std::size_t capacity) : _capacity{capacity} {}
  bool contains(std::uint64_t line) const { return _entries.count(line) != 0; }
  void touch(std::uint64_t line)
  {
    auto found = _entries.find(line);
    if (found == _entries.end()) { return; }
    _lru.splice(_lru.begin(), _lru, found->second);
    found->second = _lru.begin();
  }
  // Policy-relative counterfactual: retain incumbents when the cache is full.
  void admit_if_space(std::uint64_t line)
  {
    if (_entries.count(line) != 0 || _entries.size() == _capacity) { return; }
    _lru.push_front(line);
    _entries.emplace(line, _lru.begin());
  }

 private:
  std::size_t _capacity;
  std::list<std::uint64_t> _lru;
  std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator> _entries;
};

std::uint64_t number(char const* value) { return std::stoull(value); }
ScoreMode score_mode(std::string const& policy)
{
  if (policy == "dual_score" || policy == "dual_max") { return ScoreMode::maximum; }
  if (policy == "dual_weighted") { return ScoreMode::weighted; }
  if (policy == "dual_multiplicative") { return ScoreMode::multiplicative; }
  if (policy == "dual_lexicographic") { return ScoreMode::lexicographic; }
  return ScoreMode::none;
}
bool is_dual(std::string const& policy) { return score_mode(policy) != ScoreMode::none; }

struct NormalizedScore {
  double frequency{};
  double momentum{};
};
NormalizedScore normalize(kvikio::detail::FrequencyMomentumDecision const& decision,
                          std::uint64_t frequency_threshold,
                          std::uint64_t momentum_threshold)
{
  return {static_cast<double>(decision.frequency) / frequency_threshold,
          static_cast<double>(decision.momentum) / momentum_threshold};
}
double scalar_score(NormalizedScore const& score, ScoreMode mode)
{
  if (mode == ScoreMode::maximum) { return std::max(score.frequency, score.momentum); }
  if (mode == ScoreMode::weighted) { return 0.5 * score.frequency + 0.5 * score.momentum; }
  if (mode == ScoreMode::multiplicative) {
    return (1.0 + score.frequency) * (1.0 + score.momentum) - 1.0;
  }
  return 0.0;
}
bool candidate_beats_victim(kvikio::detail::FrequencyMomentumDecision const& candidate,
                            kvikio::detail::FrequencyMomentumDecision const& victim,
                            std::uint64_t frequency_threshold,
                            std::uint64_t momentum_threshold,
                            ScoreMode mode)
{
  auto const c = normalize(candidate, frequency_threshold, momentum_threshold);
  auto const v = normalize(victim, frequency_threshold, momentum_threshold);
  if (mode == ScoreMode::lexicographic) {
    // Momentum-first lexicographic ordering. Equal scores reject the candidate.
    return std::make_tuple(c.momentum >= 1.0, c.frequency >= 1.0, c.momentum, c.frequency) >
           std::make_tuple(v.momentum >= 1.0, v.frequency >= 1.0, v.momentum, v.frequency);
  }
  return scalar_score(c, mode) > scalar_score(v, mode);
}
RequestClass parse_class(int value)
{
  if (value < -1 || value > 3) { throw std::runtime_error("invalid request class"); }
  return static_cast<RequestClass>(value);
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
                            policy == "momentum" || policy == "hybrid" || is_dual(policy);
  auto const max_cost = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (capacity == 0 || !valid_policy || hit_ns > max_cost || fill_ns > max_cost ||
      bypass_ns > max_cost) {
    throw std::invalid_argument("invalid policy, capacity, or cost");
  }

  std::ifstream input{trace_path};
  if (!input) { throw std::runtime_error("cannot open trace"); }
  std::vector<Event> events;
  std::string row_text;
  while (std::getline(input, row_text)) {
    if (row_text.empty()) { continue; }
    std::istringstream row{row_text};
    Event event{};
    int request_class{-1};
    if (!(row >> event.line >> event.phase >> event.hot)) {
      throw std::runtime_error("malformed trace row");
    }
    if (row >> request_class) { event.request_class = parse_class(request_class); }
    else if (event.hot >= 0) {
      event.request_class = event.hot ? RequestClass::hot : RequestClass::background;
    }
    events.push_back(event);
  }
  if (events.empty()) { throw std::runtime_error("empty trace"); }

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
  ShadowCache shadow{capacity};
  std::vector<AdmissionRecord> admission_records;
  std::unordered_map<std::uint64_t, std::size_t> pending_evictions;
  std::unordered_map<std::uint64_t, std::uint64_t> burst_occurrences;
  using Episode = std::pair<std::size_t, std::uint64_t>;
  std::map<Episode, std::size_t> first_hot;
  std::map<Episode, std::size_t> detected_hot;

  auto const hit_value = static_cast<std::int64_t>(bypass_ns) -
                         static_cast<std::int64_t>(hit_ns);
  auto const admission_cost = static_cast<std::int64_t>(fill_ns) -
                              static_cast<std::int64_t>(bypass_ns);
  std::uint64_t hits{}, misses{}, admissions{}, rejected{}, evictions{};
  std::uint64_t false_admissions{}, useful_admissions{}, low_value_admissions{};
  std::uint64_t victim_reaccess_misses{}, low_value_victim_reaccess_misses{};
  std::uint64_t counterfactual_pollution_misses{}, counterfactual_saved_misses{};
  std::uint64_t score_rejections{}, completed_residencies{}, censored_residencies{};
  std::uint64_t hot_hits{}, hot_requests{}, cold_requests{}, cold_admissions{}, hot_evictions{};
  std::uint64_t scan_requests{}, scan_admissions{}, scan_replacement_attempts{};
  std::uint64_t scan_replacement_rejections{}, scan_hot_evictions{};
  std::uint64_t two_ref_second_requests{}, two_ref_second_hits{};
  std::uint64_t by_frequency{}, by_momentum{}, by_both{};

  auto finalize = [&](std::size_t admission_id, bool censored) {
    auto& record = admission_records.at(admission_id);
    if (record.finalized) { return; }
    record.finalized = true;
    censored ? ++censored_residencies : ++completed_residencies;
    if (record.hits == 0) { ++false_admissions; }
    else { ++useful_admissions; }
    auto const residency_value =
      static_cast<std::int64_t>(record.hits) * hit_value - admission_cost;
    record.low_value = residency_value <= 0;
    if (record.low_value) {
      ++low_value_admissions;
      if (record.victim_reaccessed) { ++low_value_victim_reaccess_misses; }
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
    } else if (policy == "hybrid" || is_dual(policy)) {
      decision = tracker.observe(current.line * line_size);
    }

    bool gate_admit = policy == "cache_all";
    if (policy == "frequency") { gate_admit = decision.frequency >= frequency_threshold; }
    if (policy == "momentum") { gate_admit = decision.momentum >= momentum_threshold; }
    if (policy == "hybrid" || is_dual(policy)) { gate_admit = decision.admit; }

    auto const hot = current.hot > 0;
    auto const scan = current.request_class == RequestClass::scan;
    auto const burst = current.request_class == RequestClass::burst;
    bool const second_burst_request = burst && burst_occurrences[current.line] > 0;
    if (burst) { ++burst_occurrences[current.line]; }
    if (second_burst_request) { ++two_ref_second_requests; }
    if (scan) { ++scan_requests; }
    if (current.hot >= 0) {
      hot ? ++hot_requests : ++cold_requests;
      if (hot) { first_hot.try_emplace(Episode{current.phase, current.line}, i); }
    }

    auto found = cache.find(current.line);
    bool const primary_hit = found != cache.end();
    bool const shadow_hit = shadow.contains(current.line);
    if (!primary_hit && shadow_hit) { ++counterfactual_pollution_misses; }
    if (primary_hit && !shadow_hit) { ++counterfactual_saved_misses; }
    if (shadow_hit) { shadow.touch(current.line); }
    else if (gate_admit) { shadow.admit_if_space(current.line); }

    if (primary_hit) {
      ++hits;
      if (second_burst_request) { ++two_ref_second_hits; }
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
      ++victim_reaccess_misses;
      if (cause.finalized && cause.low_value) { ++low_value_victim_reaccess_misses; }
      pending_evictions.erase(displaced);
    }

    bool admit = gate_admit;
    bool const replacement_attempt = admit && cache.size() == capacity;
    if (scan && replacement_attempt) { ++scan_replacement_attempts; }
    if (admit && is_dual(policy) && cache.size() == capacity) {
      auto const victim = lru.back();
      auto const victim_decision = tracker.estimate(victim * line_size);
      if (!candidate_beats_victim(decision, victim_decision, frequency_threshold,
                                  momentum_threshold, score_mode(policy))) {
        admit = false;
        ++score_rejections;
        if (scan) { ++scan_replacement_rejections; }
      }
    }
    if (!admit) {
      ++rejected;
      continue;
    }

    ++admissions;
    if (scan) { ++scan_admissions; }
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
      finalize(victim_entry->second.admission_id, false);
      bool const victim_hot = labeled && hot_sets[current.phase].count(victim) != 0;
      if (victim_hot) {
        ++hot_evictions;
        if (scan) { ++scan_hot_evictions; }
      }
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
  for (auto const& item : cache) { finalize(item.second.admission_id, true); }

  double detection_sum = 0;
  std::uint64_t detection_count = 0;
  if (labeled) {
    for (auto const& phase_lines : hot_sets) {
      for (auto const line_id : phase_lines.second) {
        Episode const key{phase_lines.first, line_id};
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
  auto const hit_saving_ns = static_cast<std::int64_t>(hits) * hit_value;
  auto const total_admission_cost_ns = static_cast<std::int64_t>(admissions) * admission_cost;
  auto const net_saved_ns = hit_saving_ns - total_admission_cost_ns;
  auto const counterfactual_net_misses =
    static_cast<std::int64_t>(counterfactual_saved_misses) -
    static_cast<std::int64_t>(counterfactual_pollution_misses);
  std::cout << "{\"policy\":\"" << policy << "\",\"requests\":" << events.size()
            << ",\"hits\":" << hits << ",\"misses\":" << misses
            << ",\"hit_ratio\":" << ratio(hits, events.size())
            << ",\"admissions\":" << admissions << ",\"rejected\":" << rejected
            << ",\"score_rejections\":" << score_rejections << ",\"evictions\":" << evictions
            << ",\"false_admissions\":" << false_admissions
            << ",\"false_admission_rate\":" << ratio(false_admissions, admissions)
            << ",\"low_value_admissions\":" << low_value_admissions
            << ",\"low_value_admission_rate\":" << ratio(low_value_admissions, admissions)
            << ",\"completed_residencies\":" << completed_residencies
            << ",\"censored_residencies\":" << censored_residencies
            << ",\"victim_reaccess_misses\":" << victim_reaccess_misses
            << ",\"pollution_misses\":" << victim_reaccess_misses
            << ",\"low_value_victim_reaccess_misses\":" << low_value_victim_reaccess_misses
            << ",\"counterfactual_pollution_misses\":" << counterfactual_pollution_misses
            << ",\"counterfactual_pollution_miss_rate\":"
            << ratio(counterfactual_pollution_misses, events.size())
            << ",\"counterfactual_saved_misses\":" << counterfactual_saved_misses
            << ",\"counterfactual_net_misses\":" << counterfactual_net_misses
            << ",\"shadow_policy\":\"reject_on_full\""
            << ",\"hit_saving_ns\":" << hit_saving_ns
            << ",\"admission_cost_ns\":" << total_admission_cost_ns
            << ",\"net_saved_ns\":" << net_saved_ns
            << ",\"net_saved_ns_per_request\":"
            << static_cast<double>(net_saved_ns) / events.size()
            << ",\"useful_admissions\":" << useful_admissions
            << ",\"hot_hit_ratio\":"
            << (labeled ? std::to_string(ratio(hot_hits, hot_requests)) : "null")
            << ",\"cold_admission_rate\":"
            << (labeled ? std::to_string(ratio(cold_admissions, cold_requests)) : "null")
            << ",\"hot_evictions\":" << hot_evictions
            << ",\"scan_requests\":" << scan_requests
            << ",\"scan_admissions\":" << scan_admissions
            << ",\"scan_admission_rate\":" << ratio(scan_admissions, scan_requests)
            << ",\"scan_replacement_attempts\":" << scan_replacement_attempts
            << ",\"scan_replacement_rejections\":" << scan_replacement_rejections
            << ",\"scan_hot_evictions\":" << scan_hot_evictions
            << ",\"two_ref_second_requests\":" << two_ref_second_requests
            << ",\"two_ref_second_hits\":" << two_ref_second_hits
            << ",\"two_ref_second_hit_ratio\":"
            << ratio(two_ref_second_hits, two_ref_second_requests)
            << ",\"mean_detection_delay_requests\":"
            << (labeled && detection_count != 0
                  ? std::to_string(detection_sum / detection_count) : "null")
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
            << ",\"value_model_ns\":{\"per_hit\":" << hit_value
            << ",\"per_admission_cost\":" << admission_cost << "}"
            << ",\"admission_reason\":{\"frequency\":" << by_frequency
            << ",\"momentum\":" << by_momentum << ",\"both\":" << by_both << "}}\n";
}
