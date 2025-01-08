#include <array>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <Eigen/Dense>
#include <espeak-ng/speak_lib.h>
#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
#include <soundtouch/SoundTouch.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "json.hpp"
#include "piper.hpp"
#include "utf8.h"
#include "wavfile.hpp"

namespace piper
{

#ifdef _PIPER_VERSION
// https://stackoverflow.com/questions/47346133/how-to-use-a-define-inside-a-format-string
#define _STR(x) #x
#define STR(x) _STR(x)
  const std::string VERSION = STR(_PIPER_VERSION);
#else
  const std::string VERSION = "";
#endif

  // Maximum value for 16-bit signed WAV sample
  const float MAX_WAV_VALUE = 32767.0f;

  const std::string instanceName{"piper"};

  std::string getVersion() { return VERSION; }

  // True if the string is a single UTF-8 codepoint
  bool isSingleCodepoint(std::string s)
  {
    return utf8::distance(s.begin(), s.end()) == 1;
  }

  // Get the first UTF-8 codepoint of a string
  Phoneme getCodepoint(std::string s)
  {
    utf8::iterator character_iter(s.begin(), s.begin(), s.end());
    return *character_iter;
  }

  // Load JSON config information for phonemization
  void parsePhonemizeConfig(json &configRoot, PhonemizeConfig &phonemizeConfig)
  {
    // {
    //     "espeak": {
    //         "voice": "<language code>"
    //     },
    //     "phoneme_type": "<espeak or text>",
    //     "phoneme_map": {
    //         "<from phoneme>": ["<to phoneme 1>", "<to phoneme 2>", ...]
    //     },
    //     "phoneme_id_map": {
    //         "<phoneme>": [<id1>, <id2>, ...]
    //     }
    // }

    if (configRoot.contains("espeak"))
    {
      auto espeakValue = configRoot["espeak"];
      if (espeakValue.contains("voice"))
      {
        phonemizeConfig.eSpeak.voice = espeakValue["voice"].get<std::string>();
      }
    }

    if (configRoot.contains("phoneme_type"))
    {
      auto phonemeTypeStr = configRoot["phoneme_type"].get<std::string>();
      if (phonemeTypeStr == "text")
      {
        phonemizeConfig.phonemeType = TextPhonemes;
      }
    }

    // phoneme to [id] map
    // Maps phonemes to one or more phoneme ids (required).
    if (configRoot.contains("phoneme_id_map"))
    {
      auto phonemeIdMapValue = configRoot["phoneme_id_map"];
      for (auto &fromPhonemeItem : phonemeIdMapValue.items())
      {
        std::string fromPhoneme = fromPhonemeItem.key();
        if (!isSingleCodepoint(fromPhoneme))
        {
          std::stringstream idsStr;
          for (auto &toIdValue : fromPhonemeItem.value())
          {
            PhonemeId toId = toIdValue.get<PhonemeId>();
            idsStr << toId << ",";
          }

          spdlog::error("\"{}\" is not a single codepoint (ids={})", fromPhoneme,
                        idsStr.str());
          throw std::runtime_error(
              "Phonemes must be one codepoint (phoneme id map)");
        }

        auto fromCodepoint = getCodepoint(fromPhoneme);
        for (auto &toIdValue : fromPhonemeItem.value())
        {
          PhonemeId toId = toIdValue.get<PhonemeId>();
          phonemizeConfig.phonemeIdMap[fromCodepoint].push_back(toId);
        }
      }
    }

    // phoneme to [phoneme] map
    // Maps phonemes to one or more other phonemes (not normally used).
    if (configRoot.contains("phoneme_map"))
    {
      if (!phonemizeConfig.phonemeMap)
      {
        phonemizeConfig.phonemeMap.emplace();
      }

      auto phonemeMapValue = configRoot["phoneme_map"];
      for (auto &fromPhonemeItem : phonemeMapValue.items())
      {
        std::string fromPhoneme = fromPhonemeItem.key();
        if (!isSingleCodepoint(fromPhoneme))
        {
          spdlog::error("\"{}\" is not a single codepoint", fromPhoneme);
          throw std::runtime_error(
              "Phonemes must be one codepoint (phoneme map)");
        }

        auto fromCodepoint = getCodepoint(fromPhoneme);
        for (auto &toPhonemeValue : fromPhonemeItem.value())
        {
          std::string toPhoneme = toPhonemeValue.get<std::string>();
          if (!isSingleCodepoint(toPhoneme))
          {
            throw std::runtime_error(
                "Phonemes must be one codepoint (phoneme map)");
          }

          auto toCodepoint = getCodepoint(toPhoneme);
          (*phonemizeConfig.phonemeMap)[fromCodepoint].push_back(toCodepoint);
        }
      }
    }

  } /* parsePhonemizeConfig */

  // Load JSON config for audio synthesis
  void parseSynthesisConfig(json &configRoot, SynthesisConfig &synthesisConfig)
  {
    // {
    //     "audio": {
    //         "sample_rate": 22050
    //     },
    //     "inference": {
    //         "noise_scale": 0.667,
    //         "length_scale": 1,
    //         "noise_w": 0.8,
    //         "phoneme_silence": {
    //           "<phoneme>": <seconds of silence>,
    //           ...
    //         }
    //     }
    // }

    if (configRoot.contains("audio"))
    {
      auto audioValue = configRoot["audio"];
      if (audioValue.contains("sample_rate"))
      {
        // Default sample rate is 22050 Hz
        synthesisConfig.sampleRate = audioValue.value("sample_rate", 22050);
      }
    }

    if (configRoot.contains("inference"))
    {
      // Overrides default inference settings
      auto inferenceValue = configRoot["inference"];
      if (inferenceValue.contains("noise_scale"))
      {
        synthesisConfig.noiseScale = inferenceValue.value("noise_scale", 0.667f);
      }

      if (inferenceValue.contains("length_scale"))
      {
        synthesisConfig.lengthScale = inferenceValue.value("length_scale", 1.0f);
      }

      if (inferenceValue.contains("noise_w"))
      {
        synthesisConfig.noiseW = inferenceValue.value("noise_w", 0.8f);
      }

      if (inferenceValue.contains("phoneme_silence"))
      {
        // phoneme -> seconds of silence to add after
        synthesisConfig.phonemeSilenceSeconds.emplace();
        auto phonemeSilenceValue = inferenceValue["phoneme_silence"];
        for (auto &phonemeItem : phonemeSilenceValue.items())
        {
          std::string phonemeStr = phonemeItem.key();
          if (!isSingleCodepoint(phonemeStr))
          {
            spdlog::error("\"{}\" is not a single codepoint", phonemeStr);
            throw std::runtime_error(
                "Phonemes must be one codepoint (phoneme silence)");
          }

          auto phoneme = getCodepoint(phonemeStr);
          (*synthesisConfig.phonemeSilenceSeconds)[phoneme] =
              phonemeItem.value().get<float>();
        }

      } // if phoneme_silence

    } // if inference

  } /* parseSynthesisConfig */

  void parseModelConfig(json &configRoot, ModelConfig &modelConfig)
  {

    modelConfig.numSpeakers = configRoot["num_speakers"].get<SpeakerId>();

    if (configRoot.contains("speaker_id_map"))
    {
      if (!modelConfig.speakerIdMap)
      {
        modelConfig.speakerIdMap.emplace();
      }

      auto speakerIdMapValue = configRoot["speaker_id_map"];
      for (auto &speakerItem : speakerIdMapValue.items())
      {
        std::string speakerName = speakerItem.key();
        (*modelConfig.speakerIdMap)[speakerName] =
            speakerItem.value().get<SpeakerId>();
      }
    }

  } /* parseModelConfig */

  void initialize(PiperConfig &config)
  {
    if (config.useESpeak)
    {
      // Set up espeak-ng for calling espeak_TextToPhonemesWithTerminator
      // See: https://github.com/rhasspy/espeak-ng
      spdlog::debug("Initializing eSpeak");
      int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,
                                     /*buflength*/ 0,
                                     /*path*/ config.eSpeakDataPath.c_str(),
                                     /*options*/ 0);
      if (result < 0)
      {
        throw std::runtime_error("Failed to initialize eSpeak-ng");
      }

      spdlog::debug("Initialized eSpeak");
    }

    // Load onnx model for libtashkeel
    // https://github.com/mush42/libtashkeel/
    if (config.useTashkeel)
    {
      spdlog::debug("Using libtashkeel for diacritization");
      if (!config.tashkeelModelPath)
      {
        throw std::runtime_error("No path to libtashkeel model");
      }

      spdlog::debug("Loading libtashkeel model from {}",
                    config.tashkeelModelPath.value());
      config.tashkeelState = std::make_unique<tashkeel::State>();
      tashkeel::tashkeel_load(config.tashkeelModelPath.value(),
                              *config.tashkeelState);
      spdlog::debug("Initialized libtashkeel");
    }

    spdlog::info("Initialized piper");
  }

  void terminate(PiperConfig &config)
  {
    if (config.useESpeak)
    {
      // Clean up espeak-ng
      spdlog::debug("Terminating eSpeak");
      espeak_Terminate();
      spdlog::debug("Terminated eSpeak");
    }

    spdlog::info("Terminated piper");
  }

  void loadModel(std::string modelPath, ModelSession &session, bool useCuda)
  {
    spdlog::debug("Loading onnx model from {}", modelPath);
    session.env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                           instanceName.c_str());
    session.env.DisableTelemetryEvents();

    if (useCuda)
    {
      // Use CUDA provider
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
      session.options.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Slows down performance by ~2x
    // session.options.SetIntraOpNumThreads(1);

    // Roughly doubles load time for no visible inference benefit
    // session.options.SetGraphOptimizationLevel(
    //     GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session.options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Slows down performance very slightly
    // session.options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    session.options.DisableCpuMemArena();
    session.options.DisableMemPattern();
    session.options.DisableProfiling();

    auto startTime = std::chrono::steady_clock::now();

#ifdef _WIN32
    auto modelPathW = std::wstring(modelPath.begin(), modelPath.end());
    auto modelPathStr = modelPathW.c_str();
#else
    auto modelPathStr = modelPath.c_str();
#endif

    session.onnx = Ort::Session(session.env, modelPathStr, session.options);

    auto endTime = std::chrono::steady_clock::now();
    spdlog::debug("Loaded onnx model in {} second(s)",
                  std::chrono::duration<double>(endTime - startTime).count());
  }

  // Load Onnx model and JSON config file
  void loadVoice(PiperConfig &config, std::string modelPath,
                 std::string modelConfigPath, Voice &voice,
                 std::optional<SpeakerId> &speakerId, bool useCuda)
  {
    spdlog::debug("Parsing voice config at {}", modelConfigPath);
    std::ifstream modelConfigFile(modelConfigPath);
    voice.configRoot = json::parse(modelConfigFile);

    parsePhonemizeConfig(voice.configRoot, voice.phonemizeConfig);
    parseSynthesisConfig(voice.configRoot, voice.synthesisConfig);
    parseModelConfig(voice.configRoot, voice.modelConfig);

    if (voice.modelConfig.numSpeakers > 1)
    {
      // Multi-speaker model
      if (speakerId)
      {
        voice.synthesisConfig.speakerId = speakerId;
      }
      else
      {
        // Default speaker
        voice.synthesisConfig.speakerId = 0;
      }
    }

    spdlog::debug("Voice contains {} speaker(s)", voice.modelConfig.numSpeakers);

    loadModel(modelPath, voice.session, useCuda);

  } /* loadVoice */

  // Phoneme ids to WAV audio
  void synthesize(std::vector<PhonemeId> &phonemeIds,
                  SynthesisConfig &synthesisConfig, ModelSession &session,
                  std::vector<int16_t> &audioBuffer, SynthesisResult &result)
  {
    spdlog::debug("Synthesizing audio for {} phoneme id(s)", phonemeIds.size());

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Allocate
    std::vector<int64_t> phonemeIdLengths{(int64_t)phonemeIds.size()};
    std::vector<float> scales{synthesisConfig.noiseScale,
                              synthesisConfig.lengthScale,
                              synthesisConfig.noiseW};

    std::vector<Ort::Value> inputTensors;
    std::vector<int64_t> phonemeIdsShape{1, (int64_t)phonemeIds.size()};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, phonemeIds.data(), phonemeIds.size(), phonemeIdsShape.data(),
        phonemeIdsShape.size()));

    std::vector<int64_t> phomemeIdLengthsShape{(int64_t)phonemeIdLengths.size()};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, phonemeIdLengths.data(), phonemeIdLengths.size(),
        phomemeIdLengthsShape.data(), phomemeIdLengthsShape.size()));

    std::vector<int64_t> scalesShape{(int64_t)scales.size()};
    inputTensors.push_back(
        Ort::Value::CreateTensor<float>(memoryInfo, scales.data(), scales.size(),
                                        scalesShape.data(), scalesShape.size()));

    // Add speaker id.
    // NOTE: These must be kept outside the "if" below to avoid being deallocated.
    std::vector<int64_t> speakerId{
        (int64_t)synthesisConfig.speakerId.value_or(0)};
    std::vector<int64_t> speakerIdShape{(int64_t)speakerId.size()};

    if (synthesisConfig.speakerId)
    {
      inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
          memoryInfo, speakerId.data(), speakerId.size(), speakerIdShape.data(),
          speakerIdShape.size()));
    }

    // From export_onnx.py
    std::array<const char *, 4> inputNames = {"input", "input_lengths", "scales",
                                              "sid"};
    std::array<const char *, 1> outputNames = {"output"};

    // Infer
    auto startTime = std::chrono::steady_clock::now();
    auto outputTensors = session.onnx.Run(
        Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
        inputTensors.size(), outputNames.data(), outputNames.size());
    auto endTime = std::chrono::steady_clock::now();

    if ((outputTensors.size() != 1) || (!outputTensors.front().IsTensor()))
    {
      throw std::runtime_error("Invalid output tensors");
    }
    auto inferDuration = std::chrono::duration<double>(endTime - startTime);
    result.inferSeconds = inferDuration.count();

    const float *audio = outputTensors.front().GetTensorData<float>();
    auto audioShape =
        outputTensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t audioCount = audioShape[audioShape.size() - 1];

    result.audioSeconds = (double)audioCount / (double)synthesisConfig.sampleRate;
    result.realTimeFactor = 0.0;
    if (result.audioSeconds > 0)
    {
      result.realTimeFactor = result.inferSeconds / result.audioSeconds;
    }
    spdlog::debug("Synthesized {} second(s) of audio in {} second(s)",
                  result.audioSeconds, result.inferSeconds);

    // Get max audio value for scaling
    float maxAudioValue = 0.01f;
    for (int64_t i = 0; i < audioCount; i++)
    {
      float audioValue = abs(audio[i]);
      if (audioValue > maxAudioValue)
      {
        maxAudioValue = audioValue;
      }
    }

    // We know the size up front
    audioBuffer.reserve(audioCount);

    // Scale audio to fill range and convert to int16
    float audioScale = (MAX_WAV_VALUE / std::max(0.01f, maxAudioValue));
    for (int64_t i = 0; i < audioCount; i++)
    {
      int16_t intAudioValue = static_cast<int16_t>(
          std::clamp(audio[i] * audioScale,
                     static_cast<float>(std::numeric_limits<int16_t>::min()),
                     static_cast<float>(std::numeric_limits<int16_t>::max())));

      audioBuffer.push_back(intAudioValue);
    }

    // Clean up
    for (std::size_t i = 0; i < outputTensors.size(); i++)
    {
      Ort::detail::OrtRelease(outputTensors[i].release());
    }

    for (std::size_t i = 0; i < inputTensors.size(); i++)
    {
      Ort::detail::OrtRelease(inputTensors[i].release());
    }
  }

  // ----------------------------------------------------------------------------

  // Phonemize text and synthesize audio
  void textToAudio(PiperConfig &config, Voice &voice, std::string text,
                   std::vector<int16_t> &audioBuffer, SynthesisResult &result,
                   const std::function<void()> &audioCallback)
  {

    std::size_t sentenceSilenceSamples = 0;
    if (voice.synthesisConfig.sentenceSilenceSeconds > 0)
    {
      sentenceSilenceSamples = (std::size_t)(
          voice.synthesisConfig.sentenceSilenceSeconds *
          voice.synthesisConfig.sampleRate * voice.synthesisConfig.channels);
    }

    if (config.useTashkeel)
    {
      if (!config.tashkeelState)
      {
        throw std::runtime_error("Tashkeel model is not loaded");
      }

      spdlog::debug("Diacritizing text with libtashkeel: {}", text);
      text = tashkeel::tashkeel_run(text, *config.tashkeelState);
    }

    // Phonemes for each sentence
    spdlog::debug("Phonemizing text: {}", text);
    std::vector<std::vector<Phoneme>> phonemes;

    if (voice.phonemizeConfig.phonemeType == eSpeakPhonemes)
    {
      // Use espeak-ng for phonemization
      eSpeakPhonemeConfig eSpeakConfig;
      eSpeakConfig.voice = voice.phonemizeConfig.eSpeak.voice;
      phonemize_eSpeak(text, eSpeakConfig, phonemes);
    }
    else
    {
      // Use UTF-8 codepoints as "phonemes"
      CodepointsPhonemeConfig codepointsConfig;
      phonemize_codepoints(text, codepointsConfig, phonemes);
    }

    // Synthesize each sentence independently.
    std::vector<PhonemeId> phonemeIds;
    std::map<Phoneme, std::size_t> missingPhonemes;
    for (auto phonemesIter = phonemes.begin(); phonemesIter != phonemes.end();
         ++phonemesIter)
    {
      std::vector<Phoneme> &sentencePhonemes = *phonemesIter;

      if (spdlog::should_log(spdlog::level::debug))
      {
        // DEBUG log for phonemes
        std::string phonemesStr;
        for (auto phoneme : sentencePhonemes)
        {
          utf8::append(phoneme, std::back_inserter(phonemesStr));
        }

        spdlog::debug("Converting {} phoneme(s) to ids: {}",
                      sentencePhonemes.size(), phonemesStr);
      }

      std::vector<std::shared_ptr<std::vector<Phoneme>>> phrasePhonemes;
      std::vector<SynthesisResult> phraseResults;
      std::vector<size_t> phraseSilenceSamples;

      // Use phoneme/id map from config
      PhonemeIdConfig idConfig;
      idConfig.phonemeIdMap =
          std::make_shared<PhonemeIdMap>(voice.phonemizeConfig.phonemeIdMap);

      if (voice.synthesisConfig.phonemeSilenceSeconds)
      {
        // Split into phrases
        std::map<Phoneme, float> &phonemeSilenceSeconds =
            *voice.synthesisConfig.phonemeSilenceSeconds;

        auto currentPhrasePhonemes = std::make_shared<std::vector<Phoneme>>();
        phrasePhonemes.push_back(currentPhrasePhonemes);

        for (auto sentencePhonemesIter = sentencePhonemes.begin();
             sentencePhonemesIter != sentencePhonemes.end();
             sentencePhonemesIter++)
        {
          Phoneme &currentPhoneme = *sentencePhonemesIter;
          currentPhrasePhonemes->push_back(currentPhoneme);

          if (phonemeSilenceSeconds.count(currentPhoneme) > 0)
          {
            // Split at phrase boundary
            phraseSilenceSamples.push_back(
                (std::size_t)(phonemeSilenceSeconds[currentPhoneme] *
                              voice.synthesisConfig.sampleRate *
                              voice.synthesisConfig.channels));

            currentPhrasePhonemes = std::make_shared<std::vector<Phoneme>>();
            phrasePhonemes.push_back(currentPhrasePhonemes);
          }
        }
      }
      else
      {
        // Use all phonemes
        phrasePhonemes.push_back(
            std::make_shared<std::vector<Phoneme>>(sentencePhonemes));
      }

      // Ensure results/samples are the same size
      while (phraseResults.size() < phrasePhonemes.size())
      {
        phraseResults.emplace_back();
      }

      while (phraseSilenceSamples.size() < phrasePhonemes.size())
      {
        phraseSilenceSamples.push_back(0);
      }

      // phonemes -> ids -> audio
      for (size_t phraseIdx = 0; phraseIdx < phrasePhonemes.size(); phraseIdx++)
      {
        if (phrasePhonemes[phraseIdx]->size() <= 0)
        {
          continue;
        }

        // phonemes -> ids
        phonemes_to_ids(*(phrasePhonemes[phraseIdx]), idConfig, phonemeIds,
                        missingPhonemes);
        if (spdlog::should_log(spdlog::level::debug))
        {
          // DEBUG log for phoneme ids
          std::stringstream phonemeIdsStr;
          for (auto phonemeId : phonemeIds)
          {
            phonemeIdsStr << phonemeId << ", ";
          }

          spdlog::debug("Converted {} phoneme(s) to {} phoneme id(s): {}",
                        phrasePhonemes[phraseIdx]->size(), phonemeIds.size(),
                        phonemeIdsStr.str());
        }

        // ids -> audio
        synthesize(phonemeIds, voice.synthesisConfig, voice.session, audioBuffer,
                   phraseResults[phraseIdx]);

        // Add end of phrase silence
        for (std::size_t i = 0; i < phraseSilenceSamples[phraseIdx]; i++)
        {
          audioBuffer.push_back(0);
        }

        result.audioSeconds += phraseResults[phraseIdx].audioSeconds;
        result.inferSeconds += phraseResults[phraseIdx].inferSeconds;

        phonemeIds.clear();
      }

      // Add end of sentence silence
      if (sentenceSilenceSamples > 0)
      {
        for (std::size_t i = 0; i < sentenceSilenceSamples; i++)
        {
          audioBuffer.push_back(0);
        }
      }

      if (audioCallback)
      {
        // Call back must copy audio since it is cleared afterwards.
        audioCallback();
        audioBuffer.clear();
      }

      phonemeIds.clear();
    }

    if (missingPhonemes.size() > 0)
    {
      spdlog::warn("Missing {} phoneme(s) from phoneme/id map!",
                   missingPhonemes.size());

      for (auto phonemeCount : missingPhonemes)
      {
        std::string phonemeStr;
        utf8::append(phonemeCount.first, std::back_inserter(phonemeStr));
        spdlog::warn("Missing \"{}\" (\\u{:04X}): {} time(s)", phonemeStr,
                     (uint32_t)phonemeCount.first, phonemeCount.second);
      }
    }

    if (result.audioSeconds > 0)
    {
      result.realTimeFactor = result.inferSeconds / result.audioSeconds;
    }

  } /* textToAudio */


  void speed_effect(std::vector<int16_t>& audioBuffer, float speed) {
      if (speed <= 0) {
          throw std::invalid_argument("Speed must be greater than 0");
      }

      size_t originalSize = audioBuffer.size();
      size_t newSize = static_cast<size_t>(originalSize / speed);
      std::vector<int16_t> newBuffer(newSize);

      // Resample the audio
      for (size_t i = 0; i < newSize; ++i) {
          float originalIndex = i * speed;
          size_t index = static_cast<size_t>(originalIndex);
          float fraction = originalIndex - index;

          if (index + 1 < originalSize) {
              // Linear interpolation
              newBuffer[i] = static_cast<int16_t>(
                  audioBuffer[index] * (1 - fraction) + audioBuffer[index + 1] * fraction
              );
          } else {
              newBuffer[i] = audioBuffer[index]; // Last sample
          }
      }

      // Normalize the audio
      int16_t maxAmplitude = *std::max_element(newBuffer.begin(), newBuffer.end(), [](int16_t a, int16_t b) {
          return std::abs(a) < std::abs(b);
      });

      if (maxAmplitude > 0) {
          float normalizationFactor = 32767.0f / maxAmplitude;
          for (auto& sample : newBuffer) {
              sample = static_cast<int16_t>(sample * normalizationFactor);
          }
      }

      // Replace the original buffer with the new buffer
      audioBuffer = std::move(newBuffer);
  }

  // Function to adjust volume of audio buffer
  void volume_effect(std::vector<int16_t>& audioBuffer, float volume) {
      // Validate volume range
      if (volume < -32.0f || volume > 32.0f) {
          throw std::invalid_argument("Volume parameter should be between -32 and 32.");
      }

      // Calculate the scaling factor from the volume parameter
      // Positive volume increases, negative decreases. Scale is logarithmic.
      float scaleFactor = std::pow(10.0f, volume / 20.0f);

      // Apply volume adjustment
      for (auto& sample : audioBuffer) {
          // Scale the sample and clamp it to the range of int16_t
          int32_t adjustedSample = static_cast<int32_t>(sample * scaleFactor);
          sample = static_cast<int16_t>(std::clamp(adjustedSample, -32768, 32767));
      }
  }

  void pitch_effect(std::vector<int16_t>& audioBuffer, float semitones) {
    using namespace soundtouch;

    if (semitones < -12.0f || semitones > 12.0f) {
        throw std::invalid_argument("Semitones should be within the range of -12 to 12.");
    }

    // Create SoundTouch processor
    SoundTouch soundTouch;
    soundTouch.setSampleRate(22050);
    soundTouch.setChannels(1); // Mono audio, modify if stereo
    soundTouch.setPitchSemiTones(semitones);

    // Feed data into SoundTouch
    size_t numSamples = audioBuffer.size();
    std::vector<float> floatBuffer(numSamples); // Convert to float for processing
    for (size_t i = 0; i < numSamples; ++i) {
        floatBuffer[i] = static_cast<float>(audioBuffer[i]) / 32768.0f;
    }

    soundTouch.putSamples(floatBuffer.data(), numSamples);

    // Retrieve processed samples
    std::vector<float> processedFloatBuffer;
    size_t maxBufferSize = 1024; // Chunk size for retrieval
    std::vector<float> tempBuffer(maxBufferSize);

    size_t samplesReceived;
    do {
        samplesReceived = soundTouch.receiveSamples(tempBuffer.data(), maxBufferSize);
        processedFloatBuffer.insert(processedFloatBuffer.end(), tempBuffer.begin(), tempBuffer.begin() + samplesReceived);
    } while (samplesReceived != 0);

    // Convert back to int16_t
    audioBuffer.resize(processedFloatBuffer.size());
    for (size_t i = 0; i < processedFloatBuffer.size(); ++i) {
        audioBuffer[i] = static_cast<int16_t>(std::clamp(processedFloatBuffer[i] * 32768.0f, -32768.0f, 32767.0f));
    }
  }

  // Function to compute Butterworth filter parameters
  void butter_params(double lowFreq, double highFreq, double fs, int order,
                    Eigen::VectorXd& b, Eigen::VectorXd& a) {
      if (lowFreq <= 0 || highFreq >= fs / 2 || lowFreq >= highFreq) {
          throw std::invalid_argument("Invalid frequency range for bandpass filter.");
      }

      // Normalized cutoff frequencies
      double nyquist = fs / 2.0;
      double low = lowFreq / nyquist;
      double high = highFreq / nyquist;

      // Placeholder: Replace with actual Butterworth filter design
      // Use libraries like Eigen or custom Butterworth filter implementation
      int n = 2 * order;  // Order doubled for bandpass
      b.resize(n + 1);    // Placeholder filter coefficients
      a.resize(n + 1);

      // Design coefficients (for demonstration, set to dummy values)
      for (int i = 0; i <= n; ++i) {
          b[i] = 1.0 / (n + 1);  // Dummy numerator coefficients
          a[i] = (i == 0) ? 1.0 : 0.0;  // Dummy denominator coefficients
      }
  }

  // Butterworth bandpass filter function
  std::vector<int16_t> butter_bandpass_filter(const std::vector<int16_t>& data,
                                            double lowFreq, double highFreq, double fs, int order = 5) {
      // Compute filter coefficients
      Eigen::VectorXd b, a;
      butter_params(lowFreq, highFreq, fs, order, b, a);

      // Apply the filter using convolution
      size_t dataSize = data.size();
      std::vector<int16_t> filtered(dataSize, 0.0);

      // Convolution loop (FIR/IIR filtering)
      for (size_t n = 0; n < dataSize; ++n) {
          double yn = 0.0;

          // Apply numerator (b coefficients)
          for (int i = 0; i < b.size(); ++i) {
              if (n >= i) yn += b[i] * data[n - i];
          }

          // Apply denominator (a coefficients, skip a[0])
          for (int i = 1; i < a.size(); ++i) {
              if (n >= i) yn -= a[i] * filtered[n - i];
          }

          filtered[n] = yn;
      }

      return filtered;
  }

  // Normalize audio data
  std::vector<int16_t> normalize_audio(const std::vector<int16_t>& sound,
                                      double headroom = 0.1, double maxPossibleAmp = std::pow(2.0, 15)) {
    if (sound.empty()) {
        throw std::runtime_error("Sound data is empty and cannot be normalized.");
    }

    // Find the maximum amplitude in the audio buffer
    double maxAmp = -std::numeric_limits<double>::infinity();
    for (const auto& sample : sound) {
        maxAmp = std::max(maxAmp, static_cast<double>(std::abs(sample)));  // Cast to double for comparison
    }

    // If the max amplitude is 0, the signal is silent
    if (maxAmp == 0 || maxAmp == -std::numeric_limits<double>::infinity()) {
        return sound;  // Return the original silent signal
    }

    // Calculate the target amplitude based on the headroom and maximum possible amplitude
    double targetAmp = maxPossibleAmp * std::pow(10.0, -headroom / 20.0);

    // Normalize the sound data
    std::vector<int16_t> normalizedSound = sound;
    for (auto& sample : normalizedSound) {
        sample = static_cast<int16_t>(std::clamp(static_cast<int>(sample * targetAmp / maxAmp), -32768, 32767));
    }

    return normalizedSound;
  }

  // Function to apply telephone effect
  void telephone_effect(std::vector<int16_t>& audioBuffer) {
    // Constants for telephone effect
    const bool normalize = true;
    const double lowFreq = 300.0;
    const double highFreq = 3000.0;
    const int filterOrder = 6;

    if (audioBuffer.empty()) {
        throw std::invalid_argument("Input audio buffer is empty.");
    }

    // Apply Butterworth bandpass filter
    audioBuffer = butter_bandpass_filter(audioBuffer, lowFreq, highFreq, 22050, filterOrder);

    // Normalize audio if requested
    if (normalize) {
        audioBuffer = normalize_audio(audioBuffer);
    }

    // Clamp values to the int16 range (-32768 to 32767)
    for (auto& sample : audioBuffer) {
        sample = static_cast<int16_t>(std::clamp(static_cast<int>(sample), -32768, 32767));
    }
  }

  // Function to apply cave effect
  void cave_effect(std::vector<int16_t>& audioBuffer) {
      // Apply reverb to simulate cave effect
      // This is a placeholder implementation
      for (auto& sample : audioBuffer) {
          sample = static_cast<int16_t>(sample * 0.7f);
      }
  }

  // Function to apply small cave effect
  void small_cave_effect(std::vector<int16_t>& audioBuffer) {
      // Apply reverb to simulate small cave effect
      // This is a placeholder implementation
      for (auto& sample : audioBuffer) {
          sample = static_cast<int16_t>(sample * 0.8f);
      }
  }

  // Function to apply gas mask effect
  void gas_mask_effect(std::vector<int16_t>& audioBuffer) {
      // Apply filter to simulate gas mask effect
      // This is a placeholder implementation
      for (auto& sample : audioBuffer) {
          sample = static_cast<int16_t>(sample * 0.6f);
      }
  }

  // Function to apply bad reception effect
  void bad_reception_effect(std::vector<int16_t>& audioBuffer) {
      // Apply noise to simulate bad reception effect
      // This is a placeholder implementation
      for (auto& sample : audioBuffer) {
          sample = static_cast<int16_t>(sample + (rand() % 100 - 50));
      }
  }

  // Function to apply next room effect
  void next_room_effect(std::vector<int16_t>& audioBuffer) {
      // Apply filter to simulate next room effect
      // This is a placeholder implementation
      for (auto& sample : audioBuffer) {
          sample = static_cast<int16_t>(sample * 0.9f);
      }
  }

  // Function to apply alien effect
  void alien_effect(std::vector<int16_t>& audioBuffer) {
      // Apply distortion to simulate alien effect
      // This is a placeholder implementation
      for (auto& sample : audioBuffer) {
          sample = static_cast<int16_t>(sample * 1.5f);
      }
  }

  // Function to apply alien2 effect
  void alien2_effect(std::vector<int16_t>& audioBuffer) {
      // Apply different distortion to simulate alien2 effect
      // This is a placeholder implementation
      for (auto& sample : audioBuffer) {
          sample = static_cast<int16_t>(sample * 1.2f);
      }
  }

  // Function to apply stereo effect
  void stereo_effect(std::vector<int16_t>& audioBuffer) {
      // Convert mono to stereo
      std::vector<int16_t> stereoBuffer;
      for (auto sample : audioBuffer) {
          stereoBuffer.push_back(sample);
          stereoBuffer.push_back(sample);
      }
      audioBuffer = std::move(stereoBuffer);
  }

  void applyEffects(std::vector<int16_t> &audioBuffer, AudioEffects &effects)
  {
    if (effects.speed != 1.0f)
    {
      spdlog::debug("Applying speed effect: {}", effects.speed);
      speed_effect(audioBuffer, effects.speed);
    }
    if (effects.volume != 0.0f)
    {
      spdlog::debug("Applying volume effect: {}", effects.volume);
      volume_effect(audioBuffer, effects.volume);
    }
    if (effects.semitones != 0.0f)
    {
      spdlog::debug("Applying pitch effect: {}", effects.semitones);
      pitch_effect(audioBuffer, effects.semitones);
    }
    if (effects.telephone)
    {
      spdlog::debug("Applying telephone effect");
      telephone_effect(audioBuffer);
    }
    if (effects.cave)
    {
      spdlog::debug("Applying cave effect");
      cave_effect(audioBuffer);
    }
    if (effects.smallCave)
    {
      spdlog::debug("Applying small cave effect");
      small_cave_effect(audioBuffer);
    }
    if (effects.gasMask)
    {
      spdlog::debug("Applying gas mask effect");
      gas_mask_effect(audioBuffer);
    }
    if (effects.badReception)
    {
      spdlog::debug("Applying bad reception effect");
      bad_reception_effect(audioBuffer);
    }
    if (effects.nextRoom)
    {
      spdlog::debug("Applying next room effect");
      next_room_effect(audioBuffer);
    }
    if (effects.alien)
    {
      spdlog::debug("Applying alien effect");
      alien_effect(audioBuffer);
    }
    if (effects.alien2)
    {
      spdlog::debug("Applying alien2 effect");
      alien2_effect(audioBuffer);
    }
    if (effects.stereo)
    {
      spdlog::debug("Applying stereo effect");
      stereo_effect(audioBuffer);
    }
  }


  // Phonemize text and synthesize audio to WAV file
  void textToWavFile(PiperConfig &config, Voice &voice, std::string text, AudioEffects &effects,
                     std::ostream &audioFile, SynthesisResult &result)
  {

    std::vector<int16_t> audioBuffer;
    textToAudio(config, voice, text, audioBuffer, result, NULL);

    // Apply effect
    applyEffects(audioBuffer, effects);

    // Write WAV
    auto synthesisConfig = voice.synthesisConfig;
    writeWavHeader(synthesisConfig.sampleRate, synthesisConfig.sampleWidth,
                   synthesisConfig.channels, (int32_t)audioBuffer.size(),
                   audioFile);

    audioFile.write((const char *)audioBuffer.data(),
                    sizeof(int16_t) * audioBuffer.size());

  } /* textToWavFile */

} // namespace piper
