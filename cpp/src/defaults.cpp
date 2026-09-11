/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#include <BS_thread_pool.hpp>

#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/http_status_codes.hpp>
#include <kvikio/shim/cufile.hpp>
#include <string_view>

namespace kvikio {
template <>
bool getenv_or(std::string_view env_var_name, bool default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  try {
    // Try parsing `env_var_name` as a integer
    return static_cast<bool>(std::stoi(env_val));
  } catch (std::invalid_argument const&) {
  }
  // Convert to lowercase
  std::string str{env_val};
  // Special considerations regarding the case conversion:
  // - std::tolower() is not an addressable function. Passing it to std::transform() as
  //   a function pointer, if the compile turns out successful, causes the program behavior
  //   "unspecified (possibly ill-formed)", hence the lambda. ::tolower() is addressable
  //   and does not have this problem, but the following item still applies.
  // - To avoid UB in std::tolower() or ::tolower(), the character must be cast to unsigned char.
  std::transform(
    str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
  // Trim whitespaces
  std::stringstream trimmer;
  trimmer << str;
  str.clear();
  trimmer >> str;
  // Match value
  if (str == "true" || str == "on" || str == "yes") { return true; }
  if (str == "false" || str == "off" || str == "no") { return false; }
  KVIKIO_FAIL("unknown config value " + std::string{env_var_name} + "=" + std::string{env_val},
              std::invalid_argument);
}

template <>
CompatMode getenv_or(std::string_view env_var_name, CompatMode default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  return detail::parse_compat_mode_str(env_val);
}

template <>
PolicyMode getenv_or(std::string_view env_var_name, PolicyMode default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }

  std::string value{env_val};
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  auto const first = value.find_first_not_of(" \t\n\r\f\v");
  auto const last  = value.find_last_not_of(" \t\n\r\f\v");
  value = first == std::string::npos ? std::string{} : value.substr(first, last - first + 1);

  if (value == "auto") { return PolicyMode::AUTO; }
  if (value == "host_direct") { return PolicyMode::HOST_DIRECT; }
  if (value == "host_cache") { return PolicyMode::HOST_CACHE; }
  if (value == "gds_direct") { return PolicyMode::GDS_DIRECT; }
  if (value == "gds_shaped") { return PolicyMode::GDS_SHAPED; }
  KVIKIO_FAIL("unknown config value " + std::string{env_var_name} + "=" +
                std::string{env_val},
              std::invalid_argument);
}

template <>
std::vector<int> getenv_or(std::string_view env_var_name, std::vector<int> default_val)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto* const env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return std::move(default_val); }
  std::string const int_str(env_val);
  if (int_str.empty()) { return std::move(default_val); }

  return detail::parse_http_status_codes(env_var_name, int_str);
}

unsigned int defaults::get_num_threads_from_env()
{
  KVIKIO_NVTX_FUNC_RANGE();

  auto const [env_var_name, num_threads, _] =
    getenv_or({"KVIKIO_NTHREADS", "KVIKIO_NUM_THREADS"}, 1);
  KVIKIO_EXPECT(num_threads > 0,
                std::string{env_var_name} + " has to be a positive integer",
                std::invalid_argument);
  return num_threads;
}

defaults::defaults()
{
  KVIKIO_NVTX_FUNC_RANGE();
  // Determine the default value of `compat_mode`
  {
    _compat_mode = getenv_or("KVIKIO_COMPAT_MODE", CompatMode::AUTO);
  }
  // Determine the default value of `task_size`
  {
    ssize_t const env = getenv_or("KVIKIO_TASK_SIZE", 4 * 1024 * 1024);
    KVIKIO_EXPECT(env > 0, "KVIKIO_TASK_SIZE has to be a positive integer", std::invalid_argument);
    _task_size = env;
  }
  // Determine the default value of `gds_threshold`
  {
    ssize_t const env = getenv_or("KVIKIO_GDS_THRESHOLD", 16 * 1024);
    KVIKIO_EXPECT(
      env >= 0, "KVIKIO_GDS_THRESHOLD has to be a positive integer", std::invalid_argument);
    _gds_threshold = env;
  }
  // Determine the default value of `bounce_buffer_size`
  {
    ssize_t const env = getenv_or("KVIKIO_BOUNCE_BUFFER_SIZE", 16 * 1024 * 1024);
    KVIKIO_EXPECT(
      env > 0, "KVIKIO_BOUNCE_BUFFER_SIZE has to be a positive integer", std::invalid_argument);
    _bounce_buffer_size = env;
  }
  // Host cache is opt-in. The pinned allocation is made lazily on the first eligible read.
  {
    _host_cache_enabled = getenv_or("KVIKIO_HOST_CACHE", false);
    ssize_t const capacity = getenv_or("KVIKIO_HOST_CACHE_CAPACITY", 1024 * 1024 * 1024);
    ssize_t const line_size = getenv_or("KVIKIO_HOST_CACHE_LINE_SIZE", 64 * 1024);
    ssize_t const max_io = getenv_or("KVIKIO_HOST_CACHE_MAX_IO_SIZE", 64 * 1024);
    ssize_t const region_size = getenv_or("KVIKIO_HOST_CACHE_REGION_SIZE", 1024 * 1024);
    ssize_t const admission_threshold =
      getenv_or("KVIKIO_HOST_CACHE_ADMISSION_THRESHOLD", 2);
    ssize_t const max_regions = getenv_or("KVIKIO_HOST_CACHE_MAX_REGIONS", 4096);
    KVIKIO_EXPECT(capacity > 0, "KVIKIO_HOST_CACHE_CAPACITY must be positive", std::invalid_argument);
    KVIKIO_EXPECT(line_size >= 4096 && (line_size & (line_size - 1)) == 0,
                  "KVIKIO_HOST_CACHE_LINE_SIZE must be a power of two and at least 4096",
                  std::invalid_argument);
    KVIKIO_EXPECT(max_io > 0 && max_io <= line_size,
                  "KVIKIO_HOST_CACHE_MAX_IO_SIZE must be in (0, line size]",
                  std::invalid_argument);
    KVIKIO_EXPECT(capacity >= line_size,
                  "KVIKIO_HOST_CACHE_CAPACITY must hold at least one line",
                  std::invalid_argument);
    KVIKIO_EXPECT(region_size >= line_size && region_size % line_size == 0,
                  "KVIKIO_HOST_CACHE_REGION_SIZE must contain a whole number of cache lines",
                  std::invalid_argument);
    KVIKIO_EXPECT(admission_threshold > 0 && admission_threshold <= 255,
                  "KVIKIO_HOST_CACHE_ADMISSION_THRESHOLD must be in [1, 255]",
                  std::invalid_argument);
    KVIKIO_EXPECT(max_regions > 0,
                  "KVIKIO_HOST_CACHE_MAX_REGIONS must be positive",
                  std::invalid_argument);
    _host_cache_capacity            = capacity;
    _host_cache_line_size           = line_size;
    _host_cache_max_io_size         = max_io;
    _host_cache_region_size         = region_size;
    _host_cache_admission_threshold = admission_threshold;
    _host_cache_max_regions         = max_regions;
  }
  // Request shaping is experimental and opt-in.
  {
    _request_shaping_enabled = getenv_or("KVIKIO_REQUEST_SHAPING", false);
  }
  // AUTO profiles the first requests. Other values provide controlled evaluation baselines.
  {
    _policy_mode = getenv_or("KVIKIO_POLICY_MODE", PolicyMode::AUTO);
  }
  // Determine the default value of `http_max_attempts`
  {
    ssize_t const env = getenv_or("KVIKIO_HTTP_MAX_ATTEMPTS", 3);
    KVIKIO_EXPECT(
      env > 0, "KVIKIO_HTTP_MAX_ATTEMPTS has to be a positive integer", std::invalid_argument);
    _http_max_attempts = env;
  }

  // Determine the default value of `http_timeout`
  {
    long const env = getenv_or("KVIKIO_HTTP_TIMEOUT", 60);
    KVIKIO_EXPECT(
      env > 0, "KVIKIO_HTTP_TIMEOUT has to be a positive integer", std::invalid_argument);
    _http_timeout = env;
  }

  // Determine the default value of `http_status_codes`
  {
    _http_status_codes =
      getenv_or("KVIKIO_HTTP_STATUS_CODES", std::vector<int>{429, 500, 502, 503, 504});
  }

  // Determine the default value of `auto_direct_io_read` and `auto_direct_io_write`
  {
    _auto_direct_io_read          = getenv_or("KVIKIO_AUTO_DIRECT_IO_READ", false);
    _auto_direct_io_read_overread = getenv_or("KVIKIO_AUTO_DIRECT_IO_READ_OVERREAD", false);
    _auto_direct_io_write         = getenv_or("KVIKIO_AUTO_DIRECT_IO_WRITE", true);
  }

  // Determine the default value of `thread_pool_per_block_device`
  {
    _thread_pool_per_block_device = getenv_or("KVIKIO_THREAD_POOL_PER_BLOCK_DEVICE", false);
  }
}

defaults* defaults::instance()
{
  static defaults _instance;
  return &_instance;
}
CompatMode defaults::compat_mode() { return instance()->_compat_mode; }

void defaults::set_compat_mode(CompatMode compat_mode) { instance()->_compat_mode = compat_mode; }

CompatMode defaults::infer_compat_mode_if_auto(CompatMode compat_mode) noexcept
{
  if (compat_mode == CompatMode::AUTO) {
    static auto inferred_compat_mode_for_auto = []() -> CompatMode {
      return is_cufile_available() ? CompatMode::OFF : CompatMode::ON;
    }();
    return inferred_compat_mode_for_auto;
  }
  return compat_mode;
}

bool defaults::is_compat_mode_preferred(CompatMode compat_mode) noexcept
{
  return compat_mode == CompatMode::ON ||
         (compat_mode == CompatMode::AUTO &&
          defaults::infer_compat_mode_if_auto(compat_mode) == CompatMode::ON);
}

bool defaults::is_compat_mode_preferred() { return is_compat_mode_preferred(compat_mode()); }

ThreadPool& defaults::thread_pool() { return instance()->_thread_pool; }

unsigned int defaults::thread_pool_nthreads() { return thread_pool().get_thread_count(); }

void defaults::set_thread_pool_nthreads(unsigned int nthreads)
{
  KVIKIO_EXPECT(
    nthreads > 0, "number of threads must be a positive integer", std::invalid_argument);
  thread_pool().reset(nthreads);
}

unsigned int defaults::num_threads() { return thread_pool_nthreads(); }

void defaults::set_num_threads(unsigned int nthreads) { set_thread_pool_nthreads(nthreads); }

std::size_t defaults::task_size() { return instance()->_task_size; }

void defaults::set_task_size(std::size_t nbytes)
{
  KVIKIO_EXPECT(nbytes > 0, "task size must be a positive integer", std::invalid_argument);
  instance()->_task_size = nbytes;
}

std::size_t defaults::gds_threshold() { return instance()->_gds_threshold; }

void defaults::set_gds_threshold(std::size_t nbytes) { instance()->_gds_threshold = nbytes; }

std::size_t defaults::bounce_buffer_size() { return instance()->_bounce_buffer_size; }

void defaults::set_bounce_buffer_size(std::size_t nbytes)
{
  KVIKIO_EXPECT(
    nbytes > 0, "size of the bounce buffer must be a positive integer", std::invalid_argument);
  instance()->_bounce_buffer_size = nbytes;
}

bool defaults::host_cache_enabled() { return instance()->_host_cache_enabled; }

void defaults::set_host_cache_enabled(bool enabled) { instance()->_host_cache_enabled = enabled; }

bool defaults::request_shaping_enabled() { return instance()->_request_shaping_enabled; }

void defaults::set_request_shaping_enabled(bool enabled)
{
  instance()->_request_shaping_enabled = enabled;
}

PolicyMode defaults::policy_mode() { return instance()->_policy_mode; }

void defaults::set_policy_mode(PolicyMode mode) { instance()->_policy_mode = mode; }

std::size_t defaults::host_cache_capacity() { return instance()->_host_cache_capacity; }

void defaults::set_host_cache_capacity(std::size_t nbytes)
{
  KVIKIO_EXPECT(nbytes >= host_cache_line_size(),
                "host cache capacity must hold at least one line",
                std::invalid_argument);
  instance()->_host_cache_capacity = nbytes;
}

std::size_t defaults::host_cache_line_size() { return instance()->_host_cache_line_size; }

void defaults::set_host_cache_line_size(std::size_t nbytes)
{
  KVIKIO_EXPECT(nbytes >= 4096 && (nbytes & (nbytes - 1)) == 0,
                "host cache line size must be a power of two and at least 4096",
                std::invalid_argument);
  KVIKIO_EXPECT(nbytes <= host_cache_capacity(),
                "host cache line size cannot exceed capacity",
                std::invalid_argument);
  KVIKIO_EXPECT(host_cache_max_io_size() <= nbytes,
                "host cache line size cannot be smaller than max I/O size",
                std::invalid_argument);
  instance()->_host_cache_line_size = nbytes;
}

std::size_t defaults::host_cache_max_io_size() { return instance()->_host_cache_max_io_size; }

void defaults::set_host_cache_max_io_size(std::size_t nbytes)
{
  KVIKIO_EXPECT(nbytes > 0 && nbytes <= host_cache_line_size(),
                "host cache max I/O size must be in (0, line size]",
                std::invalid_argument);
  instance()->_host_cache_max_io_size = nbytes;
}

std::size_t defaults::host_cache_region_size() { return instance()->_host_cache_region_size; }

void defaults::set_host_cache_region_size(std::size_t nbytes)
{
  auto const line_size = host_cache_line_size();
  KVIKIO_EXPECT(nbytes >= line_size && nbytes % line_size == 0,
                "host cache region size must contain a whole number of cache lines",
                std::invalid_argument);
  instance()->_host_cache_region_size = nbytes;
}

std::size_t defaults::host_cache_admission_threshold()
{
  return instance()->_host_cache_admission_threshold;
}

void defaults::set_host_cache_admission_threshold(std::size_t accesses)
{
  KVIKIO_EXPECT(accesses > 0 && accesses <= 255,
                "host cache admission threshold must be in [1, 255]",
                std::invalid_argument);
  instance()->_host_cache_admission_threshold = accesses;
}

std::size_t defaults::host_cache_max_regions() { return instance()->_host_cache_max_regions; }

void defaults::set_host_cache_max_regions(std::size_t regions)
{
  KVIKIO_EXPECT(regions > 0,
                "host cache max regions must be positive",
                std::invalid_argument);
  instance()->_host_cache_max_regions = regions;
}

std::size_t defaults::http_max_attempts() { return instance()->_http_max_attempts; }

void defaults::set_http_max_attempts(std::size_t attempts)
{
  KVIKIO_EXPECT(attempts > 0, "attempts must be a positive integer", std::invalid_argument);
  instance()->_http_max_attempts = attempts;
}

std::vector<int> const& defaults::http_status_codes() { return instance()->_http_status_codes; }

void defaults::set_http_status_codes(std::vector<int> status_codes)
{
  instance()->_http_status_codes = std::move(status_codes);
}

long defaults::http_timeout() { return instance()->_http_timeout; }
void defaults::set_http_timeout(long timeout_seconds)
{
  KVIKIO_EXPECT(
    timeout_seconds > 0, "timeout_seconds must be a positive integer", std::invalid_argument);
  instance()->_http_timeout = timeout_seconds;
}

bool defaults::auto_direct_io_read() { return instance()->_auto_direct_io_read; }

void defaults::set_auto_direct_io_read(bool flag) { instance()->_auto_direct_io_read = flag; }

bool defaults::auto_direct_io_read_overread() { return instance()->_auto_direct_io_read_overread; }

void defaults::set_auto_direct_io_read_overread(bool flag)
{
  instance()->_auto_direct_io_read_overread = flag;
}

bool defaults::auto_direct_io_write() { return instance()->_auto_direct_io_write; }

void defaults::set_auto_direct_io_write(bool flag) { instance()->_auto_direct_io_write = flag; }

bool defaults::thread_pool_per_block_device() { return instance()->_thread_pool_per_block_device; }

void defaults::set_thread_pool_per_block_device(bool flag)
{
  instance()->_thread_pool_per_block_device = flag;
}
}  // namespace kvikio
