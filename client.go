package main

import (
	"fmt"
	"net"

	"gocv.io/x/gocv"
)

var (
	deviceID = 0
	addr     = ":8080"
	protocol = "udp"
)

func main() {
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening device %v\n", deviceID)
		return
	}
	defer webcam.Close()

	img := gocv.NewMat()
	defer img.Close()

	udpConn, err := net.Dial(protocol, addr)
	if err != nil {
		fmt.Printf("Some error %v", err)
		return
	}
	defer udpConn.Close()

	fmt.Println("Listen " + protocol + " from " + udpConn.LocalAddr().String())

	if ok := webcam.Read(&img); !ok {
		fmt.Printf("Error cannot read device %v\n", deviceID)
		return
	}

	fmt.Println("Start streaming")
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}

		if img.Empty() {
			continue
		}

		fmt.Println("Buff size ", len(img.ToBytes()))

		_, err := udpConn.Write(img.ToBytes()[0:22000])
		if err != nil {
			fmt.Println("Error write: ", err)
			return
		}
	}
}
