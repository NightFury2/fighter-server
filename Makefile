dev:
	go run ./cmd/game-server/main.go

build:
	rm -rf ./game-server
	go build -v ./cmd/game-server

build_exe:
	rm -rf ./game-server.exe
	env GOOS=windows GOARCH=amd64 go build ./cmd/game-server/

run: build
	./game-server

.DEFAULT_GOAL := run