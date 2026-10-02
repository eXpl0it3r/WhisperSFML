#pragma once

#include <string>

struct Transcript
{
    // Text that won't change anymore
    std::string committed;
    // Text for the audio that is still being worked on
    std::string tentative;
};
