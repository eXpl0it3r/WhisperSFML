#include <algorithm>
#include <cstdint>
#include <future>
#include <iostream>
#include <thread>

#include <whisper.h>

#include <SFML/Audio/Sound.hpp>
#include <SFML/Audio/SoundBuffer.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/VideoMode.hpp>

std::vector<float> convertTo32BitFloat(const sf::SoundBuffer& buffer)
{
    auto* const samples = buffer.getSamples();
    auto convertedSamples = std::vector<float>{};
    convertedSamples.reserve((buffer.getSampleCount() + 1) / 2);

    if (buffer.getChannelCount() == 2)
    {
        for (auto i = 0u; i < buffer.getSampleCount(); i += 2)
        {
            convertedSamples.push_back(static_cast<float>(samples[i] + samples[i + 1]) / 65536.0f);
        }
    }
    else
    {
        for (auto i = 0u; i < buffer.getSampleCount(); ++i)
        {
            convertedSamples.push_back(static_cast<float>(samples[i]) / 65536.0f);
        }
    }

    return convertedSamples;
}

whisper_full_params setParameters()
{
    auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    parameters.n_threads = std::min(8, static_cast<std::int32_t>(std::thread::hardware_concurrency()));
    parameters.language = "en";
    parameters.offset_ms = 0;
    parameters.print_realtime = true;
    parameters.print_progress = false;
    parameters.print_timestamps = true;
    parameters.print_special = false;

    return parameters;
}

int main()
{
    auto buffer = sf::SoundBuffer{};
    if (!buffer.loadFromFile("res/George_W_Bush_Columbia_FINAL.ogg"))
    {
        std::cerr << "Failed to load sound!" << std::endl;
        return -1;
    }

    if (buffer.getChannelCount() > 2)
    {
        std::cerr << "Only supporting mono or stereo sounds" << std::endl;
        return -1;
    }

    auto convertedSamples = convertTo32BitFloat(buffer);
    auto* context = whisper_init_from_file("res/ggml-model-whisper-small.bin");

    auto processing = std::async(std::launch::async, [&context, &convertedSamples]()
    {
        const auto parameters = setParameters();
        if (whisper_full(context, parameters, convertedSamples.data(), static_cast<int>(convertedSamples.size())) != 0)
        {
            std::cerr << "Failed to process audio" << std::endl;
        }
    });

    auto window = sf::RenderWindow{sf::VideoMode{{1200, 500}}, "WhisperSFML"};
    window.setFramerateLimit(30);

    auto font = sf::Font{};
    if (!font.openFromFile("res/LinLibertine_R.ttf"))
    {
        std::cerr << "Failed to load font!" << std::endl;
        return -1;
    }

    auto text = sf::Text{font, "Hello World!", 16};
    auto string = std::string{};

    auto sound = sf::Sound{buffer};
    sound.play();

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        string = "";
        const int n_segments = whisper_full_n_segments(context);
        for (int i = 0; i < n_segments; ++i)
        {
            string += whisper_full_get_segment_text(context, i);
            string += '\n';
        }

        text.setString(string);

        window.clear();
        window.draw(text);
        window.display();
    }

    processing.get();
    whisper_free(context);
}
