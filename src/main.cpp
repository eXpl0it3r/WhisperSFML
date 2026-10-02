#include "FileTranscriber.hpp"
#include "RingBuffer.hpp"
#include "StreamRecorder.hpp"
#include "StreamTranscriber.hpp"
#include "Transcript.hpp"

#include <whisper.h>

#include <SFML/Audio/Sound.hpp>
#include <SFML/Audio/SoundBuffer.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/VideoMode.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr auto windowSize = sf::Vector2u{1200, 500};
constexpr auto margin = 10.f;
constexpr auto meterHeight = 6.f;
constexpr auto characterSize = 20u;
// Only the end of the transcript is displayed, so there's no need to wrap all of it over and over
constexpr auto maximumDisplayLength = std::size_t{4000};

sf::String toString(const std::string& text)
{
    return sf::String::fromUtf8(text.begin(), text.end());
}

std::vector<std::string> wrap(sf::Text& measure, const std::string& text, const float maximumWidth)
{
    auto lines = std::vector<std::string>{};
    auto line = std::string{};
    auto words = std::istringstream{text};

    for (auto word = std::string{}; words >> word;)
    {
        const auto candidate = line.empty() ? word : line + ' ' + word;
        measure.setString(toString(candidate));

        if (!line.empty() && measure.getLocalBounds().size.x > maximumWidth)
        {
            lines.push_back(line);
            line = word;
        }
        else
        {
            line = candidate;
        }
    }

    if (!line.empty())
    {
        lines.push_back(line);
    }

    return lines;
}

std::string join(const std::vector<std::string>& lines)
{
    auto text = std::string{};
    for (const auto& line : lines)
    {
        text += line;
        text += '\n';
    }

    return text;
}
}

int main(int argc, char* argv[])
{
    const auto* const modelPath = "res/ggml-model-whisper-base.bin";
    std::cout << "Loading model " << modelPath << ", this can take a moment..." << std::endl;

    auto contextParams = whisper_context_default_params();
    const auto context = std::unique_ptr<whisper_context, void (*)(whisper_context*)>{
        whisper_init_from_file_with_params(modelPath, contextParams),
        whisper_free};
    if (!context)
    {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    std::cout << "Model loaded" << std::endl;

    auto buffer = RingBuffer{WHISPER_SAMPLE_RATE * 30};
    auto recorder = std::optional<StreamRecorder>{};
    auto streamTranscriber = std::optional<StreamTranscriber>{};

    auto sound = sf::SoundBuffer{};
    auto playback = std::optional<sf::Sound>{};
    auto fileTranscriber = std::optional<FileTranscriber>{};

    // A sound file can be passed to transcribe it instead of the microphone
    if (argc > 1)
    {
        if (!sound.loadFromFile(argv[1]))
        {
            std::cerr << "Failed to load sound!" << std::endl;
            return -1;
        }

        fileTranscriber.emplace(context.get(), sound);
        fileTranscriber->start();
        playback.emplace(sound);

        std::cout << "Transcribing " << argv[1] << ", the text shows up below and in the window" << std::endl;
    }
    else
    {
        if (!sf::SoundRecorder::isAvailable())
        {
            std::cerr << "Audio recording is not available on this system!" << std::endl;
            return -1;
        }

        recorder.emplace(buffer);
        recorder->setChannelCount(1);
        if (!recorder->start(WHISPER_SAMPLE_RATE))
        {
            std::cerr << "Failed to start audio recording!" << std::endl;
            return -1;
        }

        streamTranscriber.emplace(context.get(), buffer);
        streamTranscriber->start();

        std::cout << "Listening on " << recorder->getDevice() << ", speak and the text shows up below and in the window"
                  << std::endl;
    }

    auto window = sf::RenderWindow{sf::VideoMode{windowSize}, "WhisperSFML"};
    window.setFramerateLimit(30);

    auto font = sf::Font{};
    if (!font.openFromFile("res/LinLibertine_R.ttf"))
    {
        std::cerr << "Failed to load font!" << std::endl;
        return -1;
    }

    auto committedText = sf::Text{font, "", characterSize};
    auto tentativeText = sf::Text{font, "", characterSize};
    tentativeText.setFillColor(sf::Color{150, 150, 150});
    auto measure = sf::Text{font, "", characterSize};

    auto meter = sf::RectangleShape{};
    meter.setPosition({0.f, static_cast<float>(windowSize.y) - meterHeight});

    const auto lineSpacing = font.getLineSpacing(characterSize);
    const auto maximumWidth = static_cast<float>(windowSize.x) - 2.f * margin;
    const auto maximumLines = static_cast<std::size_t>((static_cast<float>(windowSize.y) - 2.f * margin - meterHeight) / lineSpacing);

    auto displayed = Transcript{};
    committedText.setPosition({margin, margin});
    tentativeText.setPosition({margin, margin});
    tentativeText.setString(streamTranscriber ? "Listening..." : "Transcribing...");

    if (playback)
    {
        playback->play();
    }

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        auto transcript = streamTranscriber ? streamTranscriber->transcript() : fileTranscriber->transcript();
        if (transcript.committed != displayed.committed || transcript.tentative != displayed.tentative)
        {
            displayed = transcript;

            if (transcript.committed.size() > maximumDisplayLength)
            {
                const auto cut = transcript.committed.find(' ', transcript.committed.size() - maximumDisplayLength);
                transcript.committed.erase(0, cut);
            }

            auto committedLines = wrap(measure, transcript.committed, maximumWidth);
            const auto tentativeLines = wrap(measure, transcript.tentative, maximumWidth);

            // Scroll the oldest lines out of view
            const auto lineCount = committedLines.size() + tentativeLines.size();
            const auto excess = std::min(committedLines.size(), lineCount > maximumLines ? lineCount - maximumLines : 0);
            committedLines.erase(committedLines.begin(), committedLines.begin() + static_cast<std::ptrdiff_t>(excess));

            committedText.setString(toString(join(committedLines)));
            tentativeText.setString(toString(join(tentativeLines)));
            tentativeText.setPosition({margin, margin + static_cast<float>(committedLines.size()) * lineSpacing});
        }

        // The meter fills the width of the window at four times the silence threshold
        const auto level = streamTranscriber ? streamTranscriber->level() : 0.f;
        const auto meterWidth = std::min(level / (4.f * StreamTranscriber::silenceThreshold), 1.f) * static_cast<float>(windowSize.x);
        meter.setSize({meterWidth, meterHeight});
        meter.setFillColor(level > StreamTranscriber::silenceThreshold ? sf::Color{80, 200, 120} : sf::Color{90, 90, 90});

        window.clear();
        window.draw(committedText);
        window.draw(tentativeText);
        window.draw(meter);
        window.display();
    }
}
