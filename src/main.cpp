#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>

#include <whisper.h>

#include <SFML/Audio/Sound.hpp>
#include <SFML/Audio/SoundBuffer.hpp>
#include <SFML/Audio/SoundRecorder.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/System/Time.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/VideoMode.hpp>

bool fullDecode()
{
    auto buffer = sf::SoundBuffer{};
    if (!buffer.loadFromFile("res/George_W_Bush_Columbia_FINAL.ogg"))
    {
        std::cerr << "Failed to load sound!" << std::endl;
        return false;
    }

    auto sound = sf::Sound{ buffer };
    sound.play();

    auto* const samples = buffer.getSamples();
    auto pcmf32 = std::vector<float>{};
    pcmf32.reserve((buffer.getSampleCount() + 1) / 2);

    for (auto i = 0u; i < buffer.getSampleCount(); i += 2)
    {
        pcmf32.push_back(static_cast<float>(samples[i] + samples[i + 1]) / 65536.0f);
    }

    auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    parameters.n_threads = std::min(8, static_cast<std::int32_t>(std::thread::hardware_concurrency()));
    parameters.language = "en";
    parameters.offset_ms = 0;
    parameters.print_realtime = true;
    parameters.print_progress = false;
    parameters.print_timestamps = true;
    parameters.print_special = false;
    auto context = whisper_init_from_file("res/ggml-model-whisper-small.bin");

    if (whisper_full(context, parameters, pcmf32.data(), static_cast<int>(pcmf32.size())) == 0) {
        std::cerr << "Failed to process audio" << std::endl;
        return false;
    }

    const int n_segments = whisper_full_n_segments(context);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(context, i);
        std::cout << text << "\n";
    }

    whisper_free(context);
}

class Recorder final : public sf::SoundRecorder
{
public:
    explicit Recorder(const sf::Time bufferDuration) : m_currentPosition{ 0 }, m_bufferDuration { bufferDuration }
    {
        m_currentBufferSize = static_cast<std::size_t>(std::ceil(getChannelCount() * getSampleRate() * m_bufferDuration.asSeconds()));
    }

    ~Recorder() override
    {
        stop();
    }

    sf::SoundBuffer ReadBuffer()
    {
        std::lock_guard lock(m_mutex);
        return m_readBuffer;
    }

private:
    bool onStart() override
    {
        m_currentBufferSize = static_cast<std::size_t>(std::ceil(getChannelCount() * getSampleRate() * m_bufferDuration.asSeconds()));
        m_buffer.resize(m_currentBufferSize);
        setProcessingInterval(m_bufferDuration / 4.f);
        return true;
    }

    bool onProcessSamples(const std::int16_t* samples, std::size_t sampleCount) override
    {
        if (m_currentPosition + sampleCount >= m_buffer.size())
        {
            if (!m_readBuffer.loadFromSamples(m_buffer.data(), m_currentPosition, getChannelCount(), getSampleRate()))
            {
                m_currentPosition = 0;
                return false;
            }

            m_currentPosition = 0;
        }

        std::lock_guard lock(m_mutex);
        for (auto i = 0; i < sampleCount; ++i)
        {
            m_buffer[m_currentPosition + i] = samples[i];
        }

        m_currentPosition += sampleCount;

        return true;
    }

    void onStop() override
    {
    }

    std::mutex m_mutex;
    std::vector<std::int16_t> m_buffer;
    sf::SoundBuffer m_readBuffer;
    std::size_t m_currentPosition;
    sf::Time m_bufferDuration;
    std::size_t m_currentBufferSize;
};

int main()
{
    // if (!fullDecode())
    // {
    //     return -1;
    // }

    if (!Recorder::isAvailable())
    {
        return -1;
    }

    auto recorder = Recorder{sf::seconds(1.f)};
    recorder.setChannelCount(2);

    auto window = sf::RenderWindow{ sf::VideoMode{{400, 200}}, "WhisperSFML" };
    window.setFramerateLimit(30);

    auto font = sf::Font{};
    if (!font.loadFromFile("res/LinLibertine_R.ttf"))
    {
        std::cerr << "Failed to load font!" << std::endl;
        return -1;
    }

    auto text = sf::Text{ "Hello World!", font, 21 };

    auto recordedAudio = sf::SoundBuffer{};
    auto playback = sf::Sound{};
    auto timer = sf::Clock{};

    recorder.start();

    while (window.isOpen())
    {
        for (auto event = sf::Event{}; window.pollEvent(event);)
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
        }

        if (timer.getElapsedTime() >= sf::seconds(1.f))
        {
            playback.stop();
            recordedAudio = recorder.ReadBuffer();
            playback.setBuffer(recordedAudio);
            playback.play();
            timer.restart();
        }

        window.clear();
        window.draw(text);
        window.display();
    }

    recorder.stop();
}
