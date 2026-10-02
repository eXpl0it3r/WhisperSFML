#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

// Keeps the most recent samples of an endless stream.
// Positions are absolute sample indices since the start of the stream, so a reader keeps its own
// cursor and can read the same audio again, instead of the buffer tracking what has been consumed.
class RingBuffer
{
public:
    explicit RingBuffer(const std::size_t capacity) : m_buffer(capacity)
    {
    }

    void write(const float* samples, const std::size_t count)
    {
        {
            const auto lock = std::lock_guard{m_mutex};
            const auto capacity = m_buffer.size();

            // Only the tail fits, if more than the whole capacity arrives at once
            const auto skipped = count > capacity ? count - capacity : 0;
            const auto kept = count - skipped;
            const auto index = static_cast<std::size_t>((m_writePosition + skipped) % capacity);
            const auto firstPart = std::min(kept, capacity - index);

            std::copy_n(samples + skipped, firstPart, m_buffer.begin() + index);
            std::copy_n(samples + skipped + firstPart, kept - firstPart, m_buffer.begin());
            m_writePosition += count;
        }

        m_samplesWritten.notify_all();
    }

    // Copies everything from position up to the newest sample.
    // Returns the position of the first copied sample, which is later than requested
    // when the reader fell so far behind that the samples have been overwritten.
    std::uint64_t read(std::uint64_t position, std::vector<float>& output) const
    {
        const auto lock = std::lock_guard{m_mutex};
        const auto capacity = m_buffer.size();
        const auto oldestPosition = m_writePosition > capacity ? m_writePosition - capacity : 0;

        position = std::clamp(position, oldestPosition, m_writePosition);

        const auto count = static_cast<std::size_t>(m_writePosition - position);
        const auto index = static_cast<std::size_t>(position % capacity);
        const auto firstPart = std::min(count, capacity - index);

        output.resize(count);
        std::copy_n(m_buffer.begin() + index, firstPart, output.begin());
        std::copy_n(m_buffer.begin(), count - firstPart, output.begin() + firstPart);

        return position;
    }

    // Blocks until the stream has reached the given position or the timeout expired
    bool waitFor(const std::uint64_t position, const std::chrono::milliseconds timeout) const
    {
        auto lock = std::unique_lock{m_mutex};
        return m_samplesWritten.wait_for(lock, timeout, [this, position] { return m_writePosition >= position; });
    }

private:
    std::vector<float> m_buffer;
    std::uint64_t m_writePosition{0};
    mutable std::mutex m_mutex;
    mutable std::condition_variable m_samplesWritten;
};
