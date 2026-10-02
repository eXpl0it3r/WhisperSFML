#pragma once

#include "RingBuffer.hpp"

#include <SFML/Audio/SoundRecorder.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

// Feeds the microphone input as mono float samples into a ring buffer
class StreamRecorder final : public sf::SoundRecorder
{
public:
    explicit StreamRecorder(RingBuffer& buffer) : m_buffer(buffer)
    {
    }

    ~StreamRecorder() override
    {
        // Has to happen here, otherwise onProcessSamples can still be called on a destroyed object
        stop();
    }

private:
    bool onProcessSamples(const std::int16_t* samples, const std::size_t sampleCount) override
    {
        const auto channelCount = getChannelCount();
        const auto frameCount = sampleCount / channelCount;

        m_converted.resize(frameCount);
        for (auto frame = std::size_t{0}; frame < frameCount; ++frame)
        {
            auto sum = 0;
            for (auto channel = 0u; channel < channelCount; ++channel)
            {
                sum += samples[frame * channelCount + channel];
            }

            m_converted[frame] = static_cast<float>(sum) / (32768.f * static_cast<float>(channelCount));
        }

        m_buffer.write(m_converted.data(), m_converted.size());
        return true;
    }

    RingBuffer& m_buffer;
    std::vector<float> m_converted;
};
