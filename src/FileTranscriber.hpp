#pragma once

#include "Transcript.hpp"

#include <whisper.h>

#include <SFML/Audio/SoundBuffer.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Transcribes a whole sound on its own thread.
// Whisper gets all the audio at once and reports the segments as it works its way through.
class FileTranscriber
{
public:
    FileTranscriber(whisper_context* context, const sf::SoundBuffer& sound) : m_context(context), m_samples(convert(sound))
    {
    }

    ~FileTranscriber()
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

private:
    // Whisper needs mono at its own sample rate, the resampling uses linear interpolation
    static std::vector<float> convert(const sf::SoundBuffer& sound)
    {
        const auto* const samples = sound.getSamples();
        const auto channelCount = sound.getChannelCount();
        const auto frameCount = static_cast<std::size_t>(sound.getSampleCount() / channelCount);

        const auto frame = [&](const std::size_t index)
        {
            auto sum = 0;
            for (auto channel = 0u; channel < channelCount; ++channel)
            {
                sum += samples[index * channelCount + channel];
            }

            return static_cast<float>(sum) / (32768.f * static_cast<float>(channelCount));
        };

        const auto ratio = static_cast<double>(sound.getSampleRate()) / WHISPER_SAMPLE_RATE;
        auto converted = std::vector<float>(static_cast<std::size_t>(static_cast<double>(frameCount) / ratio));

        for (auto i = std::size_t{0}; i < converted.size(); ++i)
        {
            const auto position = static_cast<double>(i) * ratio;
            const auto index = static_cast<std::size_t>(position);
            const auto fraction = static_cast<float>(position - static_cast<double>(index));
            const auto next = std::min(index + 1, frameCount - 1);

            converted[i] = frame(index) * (1.f - fraction) + frame(next) * fraction;
        }

        return converted;
    }

    void addSegments(const int newSegmentCount)
    {
        const auto segmentCount = whisper_full_n_segments(m_context);
        for (auto i = segmentCount - newSegmentCount; i < segmentCount; ++i)
        {
            const auto text = std::string{whisper_full_get_segment_text(m_context, i)};
            const auto first = text.find_first_not_of(' ');
            if (first == std::string::npos)
            {
                continue;
            }

            std::cout << text.substr(first) << std::endl;

            const auto lock = std::lock_guard{m_mutex};
            if (!m_transcript.committed.empty())
            {
                m_transcript.committed += ' ';
            }

            m_transcript.committed += text.substr(first);
        }
    }

    void run()
    {
        auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        parameters.n_threads = std::min(8, static_cast<std::int32_t>(std::thread::hardware_concurrency()));
        parameters.language = "en";
        parameters.print_realtime = false;
        parameters.print_progress = false;
        parameters.print_timestamps = false;
        parameters.print_special = false;

        // The segments can only be read safely from Whisper's own thread while it's still working
        parameters.new_segment_callback = [](whisper_context*, whisper_state*, const int newSegmentCount, void* self)
        {
            static_cast<FileTranscriber*>(self)->addSegments(newSegmentCount);
        };
        parameters.new_segment_callback_user_data = this;

        // Don't wait for the rest of the file when shutting down
        parameters.abort_callback = [](void* running) { return !static_cast<std::atomic<bool>*>(running)->load(); };
        parameters.abort_callback_user_data = &m_running;

        if (whisper_full(m_context, parameters, m_samples.data(), static_cast<int>(m_samples.size())) != 0 && m_running)
        {
            std::cerr << "Failed to process audio" << std::endl;
        }
    }

    whisper_context* m_context;
    std::vector<float> m_samples;
    Transcript m_transcript;
    mutable std::mutex m_mutex;
    std::atomic<bool> m_running{false};
    std::thread m_thread;
};
