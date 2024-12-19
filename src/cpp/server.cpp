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

#include "json.hpp"
#include "piper.hpp"

using namespace std;
using json = nlohmann::json;

enum OutputType
{
  OUTPUT_FILE,
  OUTPUT_DIRECTORY,
  OUTPUT_STDOUT,
  OUTPUT_RAW
};

struct RunConfig
{
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

string modelPath;
void parseArgsFromJson(const json &inputJson, RunConfig &runConfig);
void rawOutputProc(vector<int16_t> &sharedAudioBuffer, mutex &mutAudio,
                   condition_variable &cvAudio, bool &audioReady,
                   bool &audioFinished);

piper::PiperConfig piperConfig;
piper::Voice voice;

int main()
{
  // Create an HTTP server instance
  httplib::Server server;

  // Define a GET route at "/"
  server.Get("/", [](const httplib::Request &req, httplib::Response &res)
             { res.set_content("Hello, World! This is a GET response.", "text/plain"); });

  // Define a POST route at "/echo"
  server.Post("/tts", [](const httplib::Request &req, httplib::Response &res)
  { 
    try {
      RunConfig runConfig;
      parseArgsFromJson(json::parse(req.body), runConfig);

      #ifdef _WIN32
        // Required on Windows to show IPA symbols
        SetConsoleOutputCP(CP_UTF8);
      #endif

      spdlog::debug("Loading voice from {} (config={})",
                    runConfig.modelPath.string(),
                    runConfig.modelConfigPath.string());

      if (modelPath != runConfig.modelPath.string())
      {
        auto startTime = chrono::steady_clock::now();
        modelPath = runConfig.modelPath.string();
        piper::loadVoice(piperConfig, runConfig.modelPath.string(),
                    runConfig.modelConfigPath.string(), voice, runConfig.speakerId,
                    runConfig.useCuda);
        auto endTime = chrono::steady_clock::now();
        spdlog::info("Loaded voice in {} second(s)",
                    chrono::duration<double>(endTime - startTime).count());
      }
      else
      {
        spdlog::info("Voice already loaded");
      }

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

      // Return input body as json format
      res.set_content(req.body, "application/json");
    } catch (const std::exception &e) {
      spdlog::error("Error processing request: {}", e.what());
      res.status = 500;
      res.set_content("Internal Server Error", "text/plain");
    }
  });

  // Start the server on port 8080
  std::cout << "Server is running on http://localhost:8080\n";
  server.listen("0.0.0.0", 8080);

  return 0;
}

void parseArgsFromJson(const json &inputJson, RunConfig &runConfig)
{
  if (inputJson.contains("modelPath"))
  {
    runConfig.modelPath = inputJson["modelPath"].get<std::string>();
  }
  if (inputJson.contains("modelConfigPath"))
  {
    runConfig.modelConfigPath = inputJson["modelConfigPath"].get<std::string>();
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
}