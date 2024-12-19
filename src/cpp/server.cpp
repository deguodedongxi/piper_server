#include "httplib.h" // Include the cpp-httplib header

int main() {
    // Create an HTTP server instance
    httplib::Server server;

    // Define a GET route at "/"
    server.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("Hello, World! This is a GET response.", "text/plain");
    });

    // Define a POST route at "/echo"
    server.Post("/tts", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("You posted: " + req.body, "text/plain");
    });

    // Start the server on port 8080
    std::cout << "Server is running on http://localhost:8080\n";
    server.listen("0.0.0.0", 8080);

    return 0;
}
