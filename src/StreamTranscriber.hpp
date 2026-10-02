#pragma once

#include "RingBuffer.hpp"
#include "Transcript.hpp"

#include <whisper.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Transcribes an endless audio stream on its own thread.
//
// The audio since the last commit forms a growing window, that is transcribed again on every step.
// As long as somebody keeps talking the result is only tentative and can still change with more context.
// Once the speaker pauses, the text is committed and the window starts over, so no word is ever cut in half.
// Without a pause the window is committed at the last segment boundary, when it reaches the maximum length.
class StreamTranscriber
{
public:
    StreamTranscriber(whisper_context* context, const RingBuffer& buffer) : m_context(context), m_buffer(buffer)
    {
    }

    ~StreamTranscriber()
    {
        m_running = false;
        if (m_thread.joinable())
        {
            m_thread.join();
        }
    }

    void start()
    {
        m_running = true;
        m_thread = std::thread{[this] { run(); }};
    }

    Transcript transcript() const
    {
        const auto lock = std::lock_guard{m_mutex};
        return m_transcript;
    }

    // Volume of the latest input, to see whether it's above the silence threshold
    float level() const
    {
        return m_level;
    }

    static constexpr auto silenceThreshold = 0.008f;

private:
    struct Segment
    {
        std::string text;
        std::int64_t start;
    };

    static constexpr auto sampleRate = std::size_t{WHISPER_SAMPLE_RATE};
    static constexpr auto stepSize = sampleRate / 2;
    static constexpr auto maximumWindowSize = sampleRate * 10;
    static constexpr auto pauseSize = sampleRate * 6 / 10;
    static constexpr auto preRollSize = sampleRate / 5;
    // Whisper refuses to process less than a second
    static constexpr auto minimumInferenceSize = sampleRate * 11 / 10;
    static constexpr auto frameSize = sampleRate / 50;
    static constexpr auto cutSearchBefore = sampleRate * 8 / 10;
    static constexpr auto cutSearchAfter = sampleRate / 5;
    // A few loud frames are rather a click or a bump than speech
    static constexpr auto minimumSpeechFrames = std::size_t{5};

    static float rootMeanSquare(const float* samples, const std::size_t count)
    {
        if (count == 0)
        {
            return 0.f;
        }

        auto sum = 0.f;
        for (auto i = std::size_t{0}; i < count; ++i)
        {
            sum += samples[i] * samples[i];
        }

        return std::sqrt(sum / static_cast<float>(count));
    }

    static std::size_t countSpeechFrames(const float* samples, const std::size_t count)
    {
        auto speechFrames = std::size_t{0};
        for (auto offset = std::size_t{0}; offset + frameSize <= count; offset += frameSize)
        {
            if (rootMeanSquare(samples + offset, frameSize) > silenceThreshold)
            {
                ++speechFrames;
            }
        }

        return speechFrames;
    }

    // Whisper's timestamps are often off by a few hundred milliseconds and usually too late,
    // so instead of trusting them, the cut is made at the quietest spot around them
    static std::size_t findQuietestFrame(const float* samples, const std::size_t count, const std::size_t position)
    {
        const auto begin = position > cutSearchBefore ? position - cutSearchBefore : 0;
        const auto end = std::min(position + cutSearchAfter, count);

        auto quietest = position;
        auto quietestLevel = std::numeric_limits<float>::max();
        for (auto offset = begin; offset + frameSize <= end; offset += frameSize)
        {
            const auto level = rootMeanSquare(samples + offset, frameSize);
            if (level < quietestLevel)
            {
                quietestLevel = level;
                quietest = offset + frameSize / 2;
            }
        }

        return quietest;
    }

    static std::string trim(const std::string& text)
    {
        const auto first = text.find_first_not_of(' ');
        const auto last = text.find_last_not_of(' ');
        return first == std::string::npos ? std::string{} : text.substr(first, last - first + 1);
    }

    static std::string join(const std::vector<Segment>& segments, const std::size_t count)
    {
        auto text = std::string{};
        for (auto i = std::size_t{0}; i < count; ++i)
        {
            if (!text.empty())
            {
                text += ' ';
            }

            text += segments[i].text;
        }

        return text;
    }

    whisper_full_params createParameters()
    {
        auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        parameters.n_threads = std::min(8, static_cast<std::int32_t>(std::thread::hardware_concurrency()));
        parameters.language = "en";
        parameters.no_context = true;
        parameters.print_realtime = false;
        parameters.print_progress = false;
        parameters.print_timestamps = false;
        parameters.print_special = false;

        // Don't wait for a running inference when shutting down
        parameters.abort_callback = [](void* running) { return !static_cast<std::atomic<bool>*>(running)->load(); };
        parameters.abort_callback_user_data = &m_running;

        return parameters;
    }

    std::vector<Segment> collectSegments() const
    {
        auto segments = std::vector<Segment>{};
        const auto segmentCount = whisper_full_n_segments(m_context);
        for (auto i = 0; i < segmentCount; ++i)
        {
            auto text = trim(whisper_full_get_segment_text(m_context, i));

            // Skip annotations for non-speech like [BLANK_AUDIO] or (music)
            if (text.empty() || text.front() == '[' || text.front() == '(')
            {
                continue;
            }

            segments.push_back({std::move(text), whisper_full_get_segment_t0(m_context, i)});
        }

        return segments;
    }

    void update(const std::string& committed, const std::string& tentative)
    {
        if (!committed.empty())
        {
            std::cout << committed << std::endl;
        }

        const auto lock = std::lock_guard{m_mutex};
        if (!committed.empty())
        {
            if (!m_transcript.committed.empty())
            {
                m_transcript.committed += ' ';
            }

            m_transcript.committed += committed;
        }

        m_transcript.tentative = tentative;
    }

    void run()
    {
        using namespace std::chrono_literals;

        const auto parameters = createParameters();
        auto window = std::vector<float>{};
        auto windowStart = std::uint64_t{0};
        auto processedUntil = std::uint64_t{0};

        while (m_running)
        {
            if (!m_buffer.waitFor(processedUntil + stepSize, 100ms))
            {
                continue;
            }

            windowStart = m_buffer.read(windowStart, window);
            const auto windowSize = window.size();
            const auto windowEnd = windowStart + windowSize;
            processedUntil = windowEnd;

            const auto levelSize = std::min(windowSize, stepSize);
            m_level = rootMeanSquare(window.data() + windowSize - levelSize, levelSize);

            // Whisper makes things up when given silence, so it never gets to see it
            if (countSpeechFrames(window.data(), windowSize) < minimumSpeechFrames)
            {
                windowStart = windowEnd - std::min(windowSize, preRollSize);
                update({}, {});
                continue;
            }

            const auto paused = windowSize >= pauseSize &&
                                countSpeechFrames(window.data() + windowSize - pauseSize, pauseSize) == 0;

            window.resize(std::max(windowSize, minimumInferenceSize), 0.f);

            if (whisper_full(m_context, parameters, window.data(), static_cast<int>(window.size())) != 0)
            {
                if (m_running)
                {
                    std::cerr << "Failed to process audio" << std::endl;
                }

                continue;
            }

            const auto segments = collectSegments();

            if (paused)
            {
                update(join(segments, segments.size()), {});
                windowStart = windowEnd - preRollSize;
            }
            else if (windowSize >= maximumWindowSize)
            {
                // Keep the last segment in the window, as its sentence most likely isn't finished yet.
                // Segment timestamps are in hundredths of a second.
                const auto lastStart = segments.empty() ? 0 : segments.back().start * static_cast<std::int64_t>(sampleRate) / 100;
                if (segments.size() > 1 && lastStart > 0 && static_cast<std::size_t>(lastStart) < windowSize)
                {
                    update(join(segments, segments.size() - 1), segments.back().text);
                    windowStart += findQuietestFrame(window.data(), windowSize, static_cast<std::size_t>(lastStart));
                }
                else
                {
                    update(join(segments, segments.size()), {});
                    windowStart = windowEnd;
                }
            }
            else
            {
                update({}, join(segments, segments.size()));
            }
        }
    }

    whisper_context* m_context;
    const RingBuffer& m_buffer;
    Transcript m_transcript;
    mutable std::mutex m_mutex;
    std::atomic<float> m_level{0.f};
    std::atomic<bool> m_running{false};
    std::thread m_thread;
};
