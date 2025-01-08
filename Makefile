.PHONY: clean docker install_deps

all: install_deps
	@echo "Building..."
	cmake -Bbuild -DCMAKE_INSTALL_PREFIX=install
	cmake --build build --config Release
	cd build && ctest --config Release
	cmake --install build

docker:
	@echo "Building docker image..."
	docker buildx build . --platform linux/amd64,linux/arm64,linux/arm/v7 --output 'type=local,dest=dist'

clean:
	@echo "Cleaning up..."
	rm -rf build install dist

install_deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y libssl-dev cmake libeigen3-dev libsoundtouch-dev
