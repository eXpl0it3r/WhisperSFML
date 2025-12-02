#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <whisper.h>

#include <SFML/Audio/SoundRecorder.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/VideoMode.hpp>

// Thread-safe ring buffer for audio samples
class RingBuffer
{
public:
    explicit RingBuffer(std::size_t capacity) : m_capacity(capacity), m_buffer(capacity)
    {
    }

    // Write samples to the ring buffer (called by audio recorder)
    void write(const std::int16_t* samples, std::size_t count)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        for (std::size_t i = 0; i < count; ++i)
        {
            m_buffer[m_writePos] = samples[i];
            m_writePos = (m_writePos + 1) % m_capacity;
            
            // If buffer is full, advance read position
            if (m_size < m_capacity)
            {
                ++m_size;
            }
            else
            {
                m_readPos = (m_readPos + 1) % m_capacity;
            }
        }
    }

    // Read samples from the ring buffer (called by whisper processor)
    std::size_t read(std::vector<float>& output, std::size_t count)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        const std::size_t samplesToRead = std::min(count, m_size);
        output.clear();
        output.reserve(samplesToRead);
        
        for (std::size_t i = 0; i < samplesToRead; ++i)
        {
            // Convert int16 to float normalized between -1.0 and 1.0
            output.push_back(static_cast<float>(m_buffer[m_readPos]) / 32768.0f);
            m_readPos = (m_readPos + 1) % m_capacity;
        }
        
        m_size -= samplesToRead;
        return samplesToRead;
    }

    std::size_t available() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_size;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_size = 0;
        m_readPos = 0;
        m_writePos = 0;
    }

private:
    std::vector<std::int16_t> m_buffer;
    std::size_t m_capacity;
    std::size_t m_size{0};
    std::size_t m_readPos{0};
    std::size_t m_writePos{0};
    mutable std::mutex m_mutex;
};

// Custom audio recorder that captures audio into a ring buffer
class AudioStreamRecorder : public sf::SoundRecorder
{
public:
    explicit AudioStreamRecorder(RingBuffer& ringBuffer) : m_ringBuffer(ringBuffer)
    {
    }

protected:
    bool onStart() override
    {
        m_ringBuffer.clear();
        return true;
    }

    bool onProcessSamples(const std::int16_t* samples, std::size_t sampleCount) override
    {
        // Convert stereo to mono if needed
        if (getChannelCount() == 2)
        {
            std::vector<std::int16_t> monoSamples;
            monoSamples.reserve(sampleCount / 2);
            
            for (std::size_t i = 0; i < sampleCount; i += 2)
            {
                monoSamples.push_back(static_cast<std::int16_t>((samples[i] + samples[i + 1]) / 2));
            }
            
            m_ringBuffer.write(monoSamples.data(), monoSamples.size());
        }
        else
        {
            m_ringBuffer.write(samples, sampleCount);
        }
        
        return true; // Continue recording
    }

private:
    RingBuffer& m_ringBuffer;
};

whisper_full_params setParameters()
{
    auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    parameters.n_threads = std::min(8, static_cast<std::int32_t>(std::thread::hardware_concurrency()));
    parameters.language = "en";
    parameters.offset_ms = 0;
    parameters.print_realtime = false;
    parameters.print_progress = false;
    parameters.print_timestamps = false;
    parameters.print_special = false;

    return parameters;
}

int main()
{
    // Check if audio recording is available
    if (!sf::SoundRecorder::isAvailable())
    {
        std::cerr << "Audio recording is not available on this system!" << std::endl;
        return -1;
    }

    // Create ring buffer for 30 seconds of audio at 16kHz (Whisper's sample rate)
    const std::size_t bufferCapacity = 16000 * 30;
    RingBuffer ringBuffer(bufferCapacity);

    // Initialize Whisper context
    auto contextParams = whisper_context_default_params();
    auto* context = whisper_init_from_file_with_params("res/ggml-large-v3.bin", contextParams);
    if (context == nullptr)
    {
        std::cerr << "Failed to initialize Whisper context!" << std::endl;
        return -1;
    }

    // Transcription result storage
    std::mutex transcriptionMutex;
    std::string transcriptionText;
    std::atomic<bool> running{true};

    // Start Whisper processing thread
    auto processingThread = std::thread([&]()
    {
        const auto parameters = setParameters();
        std::vector<float> audioChunk;
        
        // Process audio in 3-second chunks
        const std::size_t chunkSize = 16000 * 3;
        
        while (running)
        {
            // Wait until we have enough samples
            if (ringBuffer.available() >= chunkSize)
            {
                // Read chunk from ring buffer
                const std::size_t samplesRead = ringBuffer.read(audioChunk, chunkSize);
                
                if (samplesRead > 0)
                {
                    // Process with Whisper
                    if (whisper_full(context, parameters, audioChunk.data(), static_cast<int>(samplesRead)) == 0)
                    {
                        // Extract transcription
                        std::string newText;
                        const int n_segments = whisper_full_n_segments(context);
                        for (int i = 0; i < n_segments; ++i)
                        {
                            const char* text = whisper_full_get_segment_text(context, i);
                            if (text != nullptr)
                            {
                                newText += text;
                                newText += " ";
                            }
                        }
                        
                        // Update transcription text
                        if (!newText.empty())
                        {
                            std::lock_guard<std::mutex> lock(transcriptionMutex);
                            transcriptionText += newText;
                            std::cout << "Transcription: " << newText << std::endl;
                        }
                    }
                }
            }
            else
            {
                // Sleep briefly if not enough data
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    });

    // Create and start audio recorder
    AudioStreamRecorder recorder(ringBuffer);
    if (!recorder.start(16000)) // 16kHz sample rate for Whisper
    {
        std::cerr << "Failed to start audio recording!" << std::endl;
        running = false;
        processingThread.join();
        whisper_free(context);
        return -1;
    }

    std::cout << "Audio recording started. Speak into your microphone..." << std::endl;

    auto window = sf::RenderWindow{sf::VideoMode{{1200, 500}}, "WhisperSFML - Live Transcription"};
    window.setFramerateLimit(30);

    auto font = sf::Font{};
    if (!font.openFromFile("res/LinLibertine_R.ttf"))
    {
        std::cerr << "Failed to load font!" << std::endl;
        recorder.stop();
        running = false;
        processingThread.join();
        whisper_free(context);
        return -1;
    }

    auto text = sf::Text{font, "Listening... Speak into your microphone.", 16};

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        // Update display with latest transcription
        {
            std::lock_guard<std::mutex> lock(transcriptionMutex);
            text.setString(transcriptionText.empty() ? "Listening... Speak into your microphone." : transcriptionText);
        }

        window.clear();
        window.draw(text);
        window.display();
    }

    // Cleanup
    recorder.stop();
    running = false;
    processingThread.join();
    whisper_free(context);
    
    return 0;
}
