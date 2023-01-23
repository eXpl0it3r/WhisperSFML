#include <algorithm>
#include <cstdint>
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

int main()
{
    auto buffer = sf::SoundBuffer{};
    if (!buffer.loadFromFile("res/George_W_Bush_Columbia_FINAL.ogg"))
    {
        std::cerr << "Failed to load sound!" << std::endl;
        return -1;
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
        return -1;
    }

    const int n_segments = whisper_full_n_segments(context);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(context, i);
        std::cout << text << "\n";
    }

    auto window = sf::RenderWindow{ sf::VideoMode{{400, 200}}, "WhisperSFML" };
    window.setFramerateLimit(30);

    auto font = sf::Font{};
    if (!font.loadFromFile("res/LinLibertine_R.ttf"))
    {
        std::cerr << "Failed to load font!" << std::endl;
        return -1;
    }

    auto text = sf::Text{ "Hello World!", font, 21 };

    while (window.isOpen())
    {
        for (auto event = sf::Event{}; window.pollEvent(event);)
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
        }

        window.clear();
        window.draw(text);
        window.display();
    }

    whisper_free(context);
}
