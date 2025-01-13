#include "httplib.h" // Include the cpp-httplib header
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "json.hpp"
#include "piper.hpp"

using namespace std;
using json = nlohmann::json;

struct common_params {
    // server params
    std::string port           = "8080";         // server listens on this network port
    int32_t timeout_read   = 60;          // http read timeout in seconds
    int32_t timeout_write  = timeout_read; // http write timeout in seconds
    int32_t n_threads_http = -1;           // number of threads to process HTTP requests (TODO: support threadpool)
};
common_params params;

enum OutputType
{
  OUTPUT_FILE,
  OUTPUT_DIRECTORY,
  OUTPUT_STDOUT,
  OUTPUT_RAW
};



struct InitConfig {
  optional<string> port = params.port;
};

struct RunConfig {
  // The sentence you want to convert to tts
  string sentence;

  string outputFile;

  // Path to .onnx voice file
  filesystem::path modelPath;

  // Path to JSON voice config file
  filesystem::path modelConfigPath;

  // Type of output to produce.
  // Default is to write a WAV file in the current directory.
  OutputType outputType = OUTPUT_DIRECTORY;

  // Path for output
  optional<filesystem::path> outputPath = filesystem::path(".");

  // Numerical id of the default speaker (multi-speaker voices)
  optional<piper::SpeakerId> speakerId;

  // Amount of noise to add during audio generation
  optional<float> noiseScale;

  // Speed of speaking (1 = normal, < 1 is faster, > 1 is slower)
  optional<float> lengthScale;

  // Variation in phoneme lengths
  optional<float> noiseW;

  // Seconds of silence to add after each sentence
  optional<float> sentenceSilenceSeconds;

  // Path to espeak-ng data directory (default is next to piper executable)
  optional<filesystem::path> eSpeakDataPath;

  // Path to libtashkeel ort model
  // https://github.com/mush42/libtashkeel/
  optional<filesystem::path> tashkeelModelPath;

  // stdin input is lines of JSON instead of text with format:
  // {
  //   "text": str,               (required)
  //   "speaker_id": int,         (optional)
  //   "speaker": str,            (optional)
  //   "output_file": str,        (optional)
  // }
  bool jsonInput = false;

  // Seconds of extra silence to insert after a single phoneme
  optional<std::map<piper::Phoneme, float>> phonemeSilenceSeconds;

  // true to use CUDA execution provider
  bool useCuda = false;
};

void parseStartupArgs(int argc, char *argv[], InitConfig &initConfig);
void parseArgsFromJson(const json &inputJson, RunConfig&runConfig, piper::AudioEffects &effects);
// void rawOutputProc(vector<int16_t> &sharedAudioBuffer, mutex &mutAudio,
//                   condition_variable &cvAudio, bool &audioReady,
//                   bool &audioFinished);

std::mutex processingMutex;


int main(int argc, char *argv[])
{
  // Create an HTTP server instance
  httplib::Server server;

  // std::unique_ptr<httplib::Server> server;
  // Create a console sink
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::info);
  console_sink->set_pattern("[%H:%M:%S] [%^%L%$] %v");

  // Create a file sink
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/piper_log.txt", true);
  file_sink->set_level(spdlog::level::debug);
  file_sink->set_pattern("[%H:%M:%S] [%L] %v");
  
  // Create a multi-sink logger
  std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
  auto logger = std::make_shared<spdlog::logger>("piper", sinks.begin(), sinks.end());
  logger->set_level(spdlog::level::debug);
  logger->set_pattern("[%H:%M:%S] [%L] %v");
  // spdlog::flush_on(spdlog::level::debug);
  spdlog::set_default_logger(logger);

  server.set_default_headers({{"Server", "piper_server.cpp"}});
  // // set timeouts and change hostname and port
  server.set_read_timeout (params.timeout_read);
  server.set_write_timeout(params.timeout_write);
  spdlog::flush_on(spdlog::level::debug);
  InitConfig initConfig;
  parseStartupArgs(argc, argv, initConfig);


  std::string modelPath;
  piper::PiperConfig piperConfig;
  piper::Voice voice;

  spdlog::info("Starting Piper TTS Server");

  // Define a GET route at "/"
  server.Get("/", [](const httplib::Request &req, httplib::Response &res)
             { res.set_content("Hello, World! This is a GET response.", "text/plain"); });

  // Define a POST route at "/echo"
  server.Post("/tts", [&modelPath, &piperConfig, &voice](const httplib::Request &req, httplib::Response &res)
  { 
    try {
      RunConfig runConfig;
      piper::AudioEffects effects;
      // // Log Body
      // std::cout << "Request body: " << req.body << std::endl;
      parseArgsFromJson(json::parse(req.body), runConfig, effects);

      // spdlog::debug("Run Config: {}", runConfig);
      // spdlog::debug("Effects: {}", effects);
      
      #ifdef _WIN32
        // Required on Windows to show IPA symbols
        SetConsoleOutputCP(CP_UTF8);
      #endif

      // std::cout << "Model Path: " << runConfig.modelPath << std::endl;
      // std::cout << "Sentence: " << runConfig.sentence << std::endl;
      // std::cout << "Output Path: " << runConfig.outputPath.value().string() << std::endl;
      // std::cout << "Use CUDA: " << runConfig.useCuda << std::endl;

      if (modelPath != runConfig.modelPath.string())
      {
        std::lock_guard<std::mutex> lock(processingMutex);
        auto startTime = chrono::steady_clock::now();
        modelPath = runConfig.modelPath.string();
        // std::cout << "Loading voice from " << runConfig.modelPath.string() << " (config=" << runConfig.modelConfigPath.string() << ")" << std::endl;
        piper::loadVoice(piperConfig, runConfig.modelPath.string(),
                    runConfig.modelConfigPath.string(), voice, runConfig.speakerId,
                    runConfig.useCuda);
        auto endTime = chrono::steady_clock::now();
        spdlog::info("Loaded onnx model in {} second(s)", std::chrono::duration<double>(endTime - startTime).count());
      }
      // else
      // {
      //   std::cout << "Model already loaded" << std::endl;
      // }

      // Get the path to the piper executable so we can locate espeak-ng-data, etc.
      // next to it.
      #ifdef _MSC_VER
        auto exePath = []() {
          wchar_t moduleFileName[MAX_PATH] = {0};
          GetModuleFileNameW(nullptr, moduleFileName, std::size(moduleFileName));
          return filesystem::path(moduleFileName);
        }();
      #else
      #ifdef __APPLE__
        auto exePath = []() {
          char moduleFileName[PATH_MAX] = {0};
          uint32_t moduleFileNameSize = std::size(moduleFileName);
          _NSGetExecutablePath(moduleFileName, &moduleFileNameSize);
          return filesystem::path(moduleFileName);
        }();
      #else
        auto exePath = filesystem::canonical("/proc/self/exe");
      #endif
      #endif

      if (voice.phonemizeConfig.phonemeType == piper::eSpeakPhonemes) {
        spdlog::debug("Voice uses eSpeak phonemes ({})",
                      voice.phonemizeConfig.eSpeak.voice);

        if (runConfig.eSpeakDataPath) {
          // User provided path
          piperConfig.eSpeakDataPath = runConfig.eSpeakDataPath.value().string();
        } else {
          // Assume next to piper executable
          piperConfig.eSpeakDataPath =
              std::filesystem::absolute(
                  exePath.parent_path().append("espeak-ng-data"))
                  .string();

          spdlog::debug("espeak-ng-data directory is expected at {}",
                        piperConfig.eSpeakDataPath);
        }
      } else {
        // Not using eSpeak
        piperConfig.useESpeak = false;
      }

      // Enable libtashkeel for Arabic
      if (voice.phonemizeConfig.eSpeak.voice == "ar") {
        piperConfig.useTashkeel = true;
        if (runConfig.tashkeelModelPath) {
          // User provided path
          piperConfig.tashkeelModelPath =
              runConfig.tashkeelModelPath.value().string();
        } else {
          // Assume next to piper executable
          piperConfig.tashkeelModelPath =
              std::filesystem::absolute(
                  exePath.parent_path().append("libtashkeel_model.ort"))
                  .string();

          spdlog::debug("libtashkeel model is expected at {}",
                        piperConfig.tashkeelModelPath.value());
        }
      }
      
      {
        std::lock_guard<std::mutex> lock(processingMutex);
        piper::initialize(piperConfig);
      }

      
      // Scales
      if (runConfig.noiseScale) {
        voice.synthesisConfig.noiseScale = runConfig.noiseScale.value();
      }

      if (runConfig.lengthScale) {
        voice.synthesisConfig.lengthScale = runConfig.lengthScale.value();
      }

      if (runConfig.noiseW) {
        voice.synthesisConfig.noiseW = runConfig.noiseW.value();
      }

      if (runConfig.sentenceSilenceSeconds) {
        voice.synthesisConfig.sentenceSilenceSeconds =
            runConfig.sentenceSilenceSeconds.value();
      }

      spdlog::debug("Synthesis config: noiseScale={}, lengthScale={}, noiseW={}, sentenceSilenceSeconds={}",
                    voice.synthesisConfig.noiseScale, voice.synthesisConfig.lengthScale,
                    voice.synthesisConfig.noiseW, voice.synthesisConfig.sentenceSilenceSeconds);
      if (runConfig.phonemeSilenceSeconds) {
        if (!voice.synthesisConfig.phonemeSilenceSeconds) {
          // Overwrite
          voice.synthesisConfig.phonemeSilenceSeconds =
              runConfig.phonemeSilenceSeconds;
        } else {
          // Merge
          for (const auto &[phoneme, silenceSeconds] :
              *runConfig.phonemeSilenceSeconds) {
            voice.synthesisConfig.phonemeSilenceSeconds->try_emplace(
                phoneme, silenceSeconds);
          }
        }

      } // if phonemeSilenceSeconds
      if (voice.synthesisConfig.phonemeSilenceSeconds) {
        std::stringstream ss;
        for (const auto& [phoneme, silenceSeconds] : *voice.synthesisConfig.phonemeSilenceSeconds) {
          ss << phoneme << ": " << silenceSeconds << ", ";
        }
        spdlog::debug("Phoneme silence seconds: {}", ss.str());
      } else {
        spdlog::debug("Phoneme silence seconds: none");
      }

      piper::SynthesisResult result;
      {
        std::lock_guard<std::mutex> lock(processingMutex);
        if (runConfig.outputType == OUTPUT_DIRECTORY || runConfig.outputType == OUTPUT_FILE) {
          // Output audio to automatically-named WAV file in a directory
          filesystem::path outputPath = runConfig.outputPath.value();
          outputPath.append(runConfig.outputFile);

          // log name
          spdlog::debug("Output file: {}", outputPath.string());

          ofstream audioFile(outputPath.string(), ios::binary);
          piper::textToWavFile(piperConfig, voice, runConfig.sentence, effects, audioFile, result);
          // Return output path to the client as json
          json outputJson;
          outputJson["outputPath"] = runConfig.outputPath.value().string();
          outputJson["outputFile"] = runConfig.outputFile;
          res.set_content(outputJson.dump(), "application/json");
        }
        else if (runConfig.outputType == OUTPUT_STDOUT) {
          // Output audio to stdout
          piper::textToWavFile(piperConfig, voice, runConfig.sentence, effects, cout, result);

          res.set_content("Audio output to stdout", "text/plain");
        }
        else if (runConfig.outputType == OUTPUT_RAW) {
          // Raw output to stdout
          stringstream buffer;
          piper::textToWavFile(piperConfig, voice, runConfig.sentence, effects, buffer, result);
          res.set_content(buffer.str(), "audio/wav");
        }
        else {
          throw runtime_error("Invalid output type");
        }
      }

      spdlog::info("Real-time factor: {} (infer={} sec, audio={} sec)",
                  result.realTimeFactor, result.inferSeconds,
                  result.audioSeconds);
      
      
    } catch (const std::exception &e) {
      spdlog::error("Error: {}", e.what());

      // Resetting variables
      modelPath = "";
      piper::terminate(piperConfig);


      res.status = 400;
      res.set_content("Error: " + string(e.what()), "text/plain");
    }
  });

  // Start the server on port 8080
  spdlog::info("Server is running on http://localhost:{}", initConfig.port.value());
  // std::cout << "Server is running on http://localhost:" << initConfig.port.value() << std::endl;
  server.listen("0.0.0.0", stoi(initConfig.port.value()));

  return 0;
}


void printUsage(char *argv[]) {
  cerr << endl;
  cerr << "usage: " << argv[0] << " [options]" << endl;
  cerr << endl;
  cerr << "options:" << endl;
  cerr << "   -h        --help              show this message and exit" << endl;
  cerr << "   -p  PORT  --port       PORT  port to use for the server (default: 8080)" << endl;
  cerr << "   -q       --quiet              disable logging" << endl;
  cerr << "   --debug                       print DEBUG messages to the console" << endl;
  cerr << endl;
}

void ensureArg(int argc, char *argv[], int argi) {
  if ((argi + 1) >= argc) {
    printUsage(argv);
    exit(0);
  }
}

// Parse command-line arguments
void parseStartupArgs(int argc, char *argv[], InitConfig &initConfig) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--port" || arg == "-p") {
      ensureArg(argc, argv, i);
      initConfig.port = argv[++i];
    }
    else if (arg == "--debug") {
      // Set DEBUG logging
      spdlog::set_level(spdlog::level::debug);
    }
    else if (arg == "-q" || arg == "--quiet") {
      // diable logging
      spdlog::set_level(spdlog::level::off);
    }
    else if (arg == "-h" || arg == "--help") {
      printUsage(argv);
      exit(0);
    }
  }
}

void parseArgsFromJson(const json &inputJson, RunConfig &runConfig, piper::AudioEffects &effects)
{
  if (inputJson.contains("sentence"))
  {
    runConfig.sentence = inputJson["sentence"].get<std::string>();
  }
  if (inputJson.contains("modelPath"))
  {
    runConfig.modelPath = inputJson["modelPath"].get<std::string>();
  }

  // Check if model path exists
  if (!filesystem::exists(runConfig.modelPath))
  {
    std::cout << "Model path does not exist: " << runConfig.modelPath.string() << std::endl;
    throw std::runtime_error("Model path does not exist: " + runConfig.modelPath.string());
  }
  // else {
  //   std::cout << "Model path exists: " << runConfig.modelPath.string() << std::endl;
  // }

  if (inputJson.contains("modelConfigPath"))
  {
    runConfig.modelConfigPath = inputJson["modelConfigPath"].get<std::string>();
  }
  else {
    runConfig.modelConfigPath = runConfig.modelPath.string() + ".json";
  }

  // Verify model config path exists
  if (!filesystem::exists(runConfig.modelConfigPath))
  {
    std::cout << "Model config path does not exist: " << runConfig.modelConfigPath.string() << std::endl;
    throw std::runtime_error("Model config path does not exist: " + runConfig.modelConfigPath.string());
  }
  // else {
  //   std::cout << "Model config path exists: " << runConfig.modelConfigPath.string() << std::endl;
  // }

  if (inputJson.contains("output_file"))
  {
    runConfig.outputFile = inputJson["output_file"].get<std::string>() + ".wav";
  }
  else {
    //TODO: generate random uuid for output file 
    runConfig.outputFile = "output.wav";
  }

  if (inputJson.contains("outputType"))
  {
    std::string outputTypeStr = inputJson["outputType"].get<std::string>();
    if (outputTypeStr == "OUTPUT_FILE")
    {
      runConfig.outputType = OUTPUT_FILE;
    }
    else if (outputTypeStr == "OUTPUT_DIRECTORY")
    {
      runConfig.outputType = OUTPUT_DIRECTORY;
    }
    else if (outputTypeStr == "OUTPUT_STDOUT")
    {
      runConfig.outputType = OUTPUT_STDOUT;
    }
    else if (outputTypeStr == "OUTPUT_RAW")
    {
      runConfig.outputType = OUTPUT_RAW;
    }
  }
  if (inputJson.contains("outputPath"))
  {
    runConfig.outputPath = inputJson["outputPath"].get<std::string>();
  }
  else {
    runConfig.outputPath = filesystem::path(".");
  }

  // Check if output path exists
  if (!filesystem::exists(runConfig.outputPath.value()))
  {
    std::cout << "Output path does not exist: " << runConfig.outputPath.value().string() << std::endl;
    throw std::runtime_error("Output path does not exist: " + runConfig.outputPath.value().string());
  }
  // else {
  //   std::cout << "Output path exists: " << runConfig.outputPath.value().string() << std::endl;
  // }



  if (inputJson.contains("speakerId"))
  {
    runConfig.speakerId = inputJson["speakerId"].get<piper::SpeakerId>();
  }
  if (inputJson.contains("noiseScale"))
  {
    runConfig.noiseScale = inputJson["noiseScale"].get<float>();
  }
  if (inputJson.contains("lengthScale"))
  {
    runConfig.lengthScale = inputJson["lengthScale"].get<float>();
  }
  if (inputJson.contains("noiseW"))
  {
    runConfig.noiseW = inputJson["noiseW"].get<float>();
  }
  if (inputJson.contains("sentenceSilenceSeconds"))
  {
    runConfig.sentenceSilenceSeconds = inputJson["sentenceSilenceSeconds"].get<float>();
  }
  if (inputJson.contains("eSpeakDataPath"))
  {
    runConfig.eSpeakDataPath = inputJson["eSpeakDataPath"].get<std::string>();
  }
  if (inputJson.contains("tashkeelModelPath"))
  {
    runConfig.tashkeelModelPath = inputJson["tashkeelModelPath"].get<std::string>();
  }
  if (inputJson.contains("jsonInput"))
  {
    runConfig.jsonInput = inputJson["jsonInput"].get<bool>();
  }
  if (inputJson.contains("phonemeSilenceSeconds"))
  {
    runConfig.phonemeSilenceSeconds = inputJson["phonemeSilenceSeconds"].get<std::map<piper::Phoneme, float>>();
  }
  if (inputJson.contains("useCuda"))
  {
    runConfig.useCuda = inputJson["useCuda"].get<bool>();
  }
  if (inputJson.contains("semitones"))
  {
    effects.semitones = inputJson["semitones"].get<float>();
  }
  if (inputJson.contains("speed"))
  {
    effects.speed = inputJson["speed"].get<float>();
  }
  if (inputJson.contains("volume"))
  {
    effects.volume = inputJson["volume"].get<float>();
  }
  if (inputJson.contains("voiceImprovement"))
  {
    effects.voiceImprovement = inputJson["voiceImprovement"].get<bool>();
  }
  if (inputJson.contains("highFramerate"))
  {
    effects.highFramerate = inputJson["highFramerate"].get<bool>();
  }
  if (inputJson.contains("telephone"))
  {
    effects.telephone = inputJson["telephone"].get<bool>();
  }
  if (inputJson.contains("cave"))
  {
    effects.cave = inputJson["cave"].get<bool>();
  }
  if (inputJson.contains("smallCave"))
  {
    effects.smallCave = inputJson["smallCave"].get<bool>();
  }
  if (inputJson.contains("gasMask"))
  {
    effects.gasMask = inputJson["gasMask"].get<bool>();
  }
  if (inputJson.contains("badReception"))
  {
    effects.badReception = inputJson["badReception"].get<bool>();
  }
  if (inputJson.contains("nextRoom"))
  {
    effects.nextRoom = inputJson["nextRoom"].get<bool>();
  }
  if (inputJson.contains("alien"))
  {
    effects.alien = inputJson["alien"].get<bool>();
  }
  if (inputJson.contains("alien2"))
  {
    effects.alien2 = inputJson["alien2"].get<bool>();
  }
  if (inputJson.contains("stereo"))
  {
    effects.stereo = inputJson["stereo"].get<bool>();
  }
}