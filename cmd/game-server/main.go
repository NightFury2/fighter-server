package main

import (
	"fmt"
	"image"
	"image/color"
	"net"
	"path/filepath"

	"gocv.io/x/gocv"
)

var (
	gocvNet *gocv.Net
	images  chan *gocv.Mat
	poses   chan [][]image.Point
	pose    [][]image.Point

	deviceID = 0
	proto    = "./assets/deploy.prototxt"
	model    = "./assets/res10_300x300_ssd_iter_140000.caffemodel"
	backend  = gocv.NetBackendDefault
	target   = gocv.NetTargetCPU

	addr     = ":8081"
	protocol = "udp"
)

func main() {
	//Build the address
	udpAddr, err := net.ResolveUDPAddr(protocol, addr)
	if err != nil {
		fmt.Println("Wrong Address")
		return
	}

	//Create the connection
	udpConn, err := net.ListenUDP(protocol, udpAddr)
	if err != nil {
		fmt.Println(err)
	}
	defer udpConn.Close()

	fmt.Println("Reading " + protocol + " from " + udpAddr.String())

	img := gocv.NewMat()
	defer img.Close()

	n := gocv.ReadNet(model, proto)
	gocvNet = &n
	if gocvNet.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, proto)
		return
	}
	defer gocvNet.Close()

	gocvNet.SetPreferableBackend(gocv.NetBackendType(backend))
	gocvNet.SetPreferableTarget(gocv.NetTargetType(target))

	var ratio float64
	var mean gocv.Scalar
	var swapRGB bool

	if filepath.Ext(model) == ".caffemodel" {
		ratio = 1.0
		mean = gocv.NewScalar(104, 177, 123, 0)
		swapRGB = false
	} else {
		ratio = 1.0 / 127.5
		mean = gocv.NewScalar(127.5, 127.5, 127.5, 0)
		swapRGB = true
	}

	udpReqCount := 0
	for {
		buf := make([]byte, 500000)

		len, remoteAddr, err := udpConn.ReadFromUDP(buf)
		if err != nil {
			fmt.Println("Error reading ", err)
			return
		}
		udpReqCount++

		// TODO implement pose detection
		img2, err := gocv.NewMatFromBytes(img.Rows(), img.Cols(), img.Type(), buf)
		if err != nil {
			fmt.Println("Error parse buff to img ", err)
			continue
		}

		// convert image Mat DataType to CV_32F
		imgCV32F := img2.Clone()
		imgCV32F.ConvertTo(&imgCV32F, gocv.MatTypeCV32F)

		// convert image Mat to 300x300 blob that the object detector can analyze
		blob := gocv.BlobFromImage(imgCV32F, ratio, image.Pt(300, 300), mean, swapRGB, false)

		// feed the blob into the detector
		gocvNet.SetInput(blob, "")

		// run a forward pass thru the network
		prob := gocvNet.Forward("")

		performDetection(&img, prob)

		prob.Close()
		blob.Close()

		// fmt.Println(results)

		udpConn.WriteToUDP(buf, remoteAddr)
		fmt.Println("Received", " count ", udpReqCount, " buff size ", len, " connection ", remoteAddr.String(), " message ", string(buf[0:20]))
	}
}

// performDetection analyzes the results from the detector network,
// which produces an output blob with a shape 1x1xNx7
// where N is the number of detections, and each detection
// is a vector of float values
// [batchId, classId, confidence, left, top, right, bottom]
func performDetection(frame *gocv.Mat, results gocv.Mat) {
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if confidence > 0.5 {
			left := int(results.GetFloatAt(0, i+3) * float32(frame.Cols()))
			top := int(results.GetFloatAt(0, i+4) * float32(frame.Rows()))
			right := int(results.GetFloatAt(0, i+5) * float32(frame.Cols()))
			bottom := int(results.GetFloatAt(0, i+6) * float32(frame.Rows()))
			gocv.Rectangle(frame, image.Rect(left, top, right, bottom), color.RGBA{0, 255, 0, 0}, 2)
		}
	}
}
