# WhisperSFML

Like the name WhisperSFML combines the OpenAI Whisper via Whisper.cpp and SFML to
demonstrate real-time audio transcription (Speak-To-Text) as well as translation.

## How To Use

- Get the WhisperSFML source code
- Make sure CMake and a compiler is installed
- Get one of [model files in the ggml format](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
  - *I recommend at least base or small*
- Replace the model filename in `src/main.cpp`
- Build and run
  - *I recommend to run it in release mode*

Note that quantized models created with an old version of Whisper.cpp only produce garbage, so make sure to get a recent one.

## Modes

WhisperSFML needs to be started from the directory containing the `res` folder and can be run in two ways.

### Microphone

```
WhisperSFML
```

Without any arguments the default recording device is transcribed while you speak.

The white text is final, while the gray text is still being worked on and can change as more audio comes in.
The text is finalized as soon as you make a short pause, and it's additionally printed to the console.
The bar at the bottom shows the input level and turns green, when it's loud enough to count as speech.
If it never turns green with your microphone, lower `silenceThreshold` in `src/StreamTranscriber.hpp`.

### Sound File

```
WhisperSFML res/George_W_Bush_Columbia_FINAL.ogg
```

When a sound file is passed, it's played back and transcribed as a whole.
Note that Whisper is usually quicker than the playback, so the text tends to run ahead of what you hear.

![Screenshot of WhisperSFML in action](https://github.com/user-attachments/assets/585c8804-7104-4a1f-8caa-2625048bb787)

## Resources

- [OpenAI Whisper](https://openai.com/blog/whisper/)
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [SFML](https://github.com/SFML/SFML)

## License

The code itself is is available under 2 licenses: Public Domain or MIT -- choose whichever you prefer, see also the license file.