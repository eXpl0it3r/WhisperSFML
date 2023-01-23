#include <iostream>

#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/VideoMode.hpp>

int main()
{
    auto window = sf::RenderWindow{ sf::VideoMode{{400, 200}}, "WhisperSFML" };
    window.setFramerateLimit(30);

    auto font = sf::Font{};
    if (!font.loadFromFile("res/LinLibertine_R.ttf"))
    {
        std::cerr << "Failed to load font!" << std::endl;
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
}
