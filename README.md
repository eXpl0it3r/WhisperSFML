# WhisperSFML

Like the name WhisperSFML combines the OpenAI Whisper via Whisper.cpp and SFML to
demonstrate real-time audio transcription (Speak-To-Text) as well as translation.

## How To Use

- Get the WhisperSFML source code
- Make sure CMake and a compiler is installed
- Get one of [model files in the ggml format](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
  - *I recommend at least base or small*
- (optional) Pick some sound file of your desire
- Replace the filenames in the source files
- Build and run
  - *I recommend to run it in release mode*

![Screenshot of WhisperSFML in action](https://user-images.githubusercontent.com/920861/214178232-71101c23-874f-41b0-836b-2235d366c11c.png)

## Resources

- [OpenAI Whisper](https://openai.com/blog/whisper/)
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [SFML](https://github.com/SFML/SFML)

## License

The code itself is is available under 2 licenses: Public Domain or MIT -- choose whichever you prefer, see also the license file.