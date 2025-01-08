.PHONY: clean docker install_deps

all: install_deps
	@echo "Building..."
	cmake -Bbuild -DCMAKE_INSTALL_PREFIX=install
	cmake --build build --config Release
	cd build
	ctest
	cmake --install build

docker:
	@echo "Building docker image..."
	docker buildx build . --platform linux/amd64,linux/arm64,linux/arm/v7 --output 'type=local,dest=dist'

clean:
	@echo "Cleaning up..."
	rm -rf build install dist

install_deps:
	@echo "Installing dependencies..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "Invoke-WebRequest -Uri 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip' -OutFile 'eigen-3.4.0.zip'"
	@powershell -Command "Expand-Archive -Path 'eigen-3.4.0.zip' -DestinationPath '.'"
	@powershell -Command "Move-Item -Path 'eigen-3.4.0' -Destination 'eigen3'"
	@powershell -Command "Invoke-WebRequest -Uri 'https://gitlab.com/soundtouch/soundtouch/-/archive/2.3.1/soundtouch-2.3.1.zip' -OutFile 'soundtouch-2.3.1.zip'"
	@powershell -Command "Expand-Archive -Path 'soundtouch-2.3.1.zip' -DestinationPath '.'"
	@powershell -Command "cd soundtouch-2.3.1; mkdir build; cd build; cmake ..; cmake --build . --config Release; cmake --install . --prefix ../../soundtouch"
else
	sudo apt-get update
	sudo apt-get install -y libssl-dev cmake libeigen3-dev libsoundtouch-dev
endif
