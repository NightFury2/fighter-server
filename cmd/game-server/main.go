package main

import (
	"encoding/json"
	"fmt"
	"image"
	"log"
	"net"

	"gocv.io/x/gocv"
)

var (
	gocvNet *gocv.Net
	images  chan *gocv.Mat
	poses   chan [][]image.Point
	pose    [][]image.Point

	deviceID = 0
	proto    = "./assets/openpose_pose_coco.prototxt"
	model    = "./assets/pose_iter_440000.caffemodel"
	backend  = gocv.NetBackendDefault
	target   = gocv.NetTargetCPU

	addr     = ":8080"
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

	n := gocv.ReadNet(model, proto)
	gocvNet = &n
	if gocvNet.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, proto)
		return
	}
	defer gocvNet.Close()

	gocvNet.SetPreferableBackend(gocv.NetBackendType(backend))
	gocvNet.SetPreferableTarget(gocv.NetTargetType(target))

	udpReqCount := 0
	for {
		buf := make([]byte, 500000)

		len, remoteAddr, err := udpConn.ReadFromUDP(buf)
		if err != nil {
			fmt.Println("Error reading ", err)
			return
		}
		udpReqCount++
		fmt.Println("Received", " count ", udpReqCount, " buff size ", len, " connection ", remoteAddr.String(), " message ", string(buf[0:5]))

		frame, err := gocv.IMDecode(buf, gocv.IMReadColor)
		if err != nil {
			fmt.Println("Error parse buff to img ", err)
			continue
		}

		if frame.Empty() {
			fmt.Println("Image is empty")
			continue
		}

		blob := gocv.BlobFromImage(frame, 1.0, image.Pt(224, 224), gocv.NewScalar(0, 0, 0, 0), false, false)
		if blob.Empty() {
			fmt.Println("Invalid blob in Caffe test")
		}

		// feed the blob into the detector
		gocvNet.SetInput(blob, "")

		// run a forward pass thru the network
		prob := gocvNet.Forward("")

		var midx int
		s := prob.Size()
		nparts, h, w := s[1], s[2], s[3]

		// find out, which model we have
		switch nparts {
		case 19:
			// COCO body
			midx = 0
			nparts = 18 // skip background
		case 16:
			// MPI body
			midx = 1
			nparts = 15 // skip background
		case 22:
			// hand
			midx = 2
		default:
			fmt.Println("there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand model")
			return
		}

		// find the most likely match for each part
		pts := make([]image.Point, 22)
		for i := 0; i < nparts; i++ {
			pts[i] = image.Pt(-1, -1)
			heatmap, _ := prob.FromPtr(h, w, gocv.MatTypeCV32F, 0, i)

			_, maxVal, _, maxLoc := gocv.MinMaxLoc(heatmap)
			if maxVal > 0.1 {
				pts[i] = maxLoc
			}
			heatmap.Close()
		}

		// determine scale factor
		sX := int(float32(frame.Cols()) / float32(w))
		sY := int(float32(frame.Rows()) / float32(h))

		// create the results array of pairs of points with the lines that best fit
		// each body part, e.g.
		// [[point A for body part 1, point B for body part 1],
		//  [point A for body part 2, point B for body part 2], ...]
		results := [][]image.Point{}
		for _, p := range PosePairs[midx] {
			a := pts[p[0]]
			b := pts[p[1]]

			// high enough confidence in this pose?
			if a.X <= 0 || a.Y <= 0 || b.X <= 0 || b.Y <= 0 {
				continue
			}

			// scale to image size
			a.X *= sX
			a.Y *= sY
			b.X *= sX
			b.Y *= sY

			results = append(results, []image.Point{a, b})
		}
		prob.Close()
		blob.Close()
		frame.Close()

		var jsonData []byte
		fmt.Println(results)

		jsonData, err = json.Marshal(results)
		if err != nil {
			log.Println(err)
		}

		fmt.Println(jsonData)

		udpConn.WriteToUDP(jsonData, remoteAddr)
	}
}

// PosePairs is a table of the body part connections in the format [model_id][pair_id][from/to]
// For details please see:
// https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
//
var PosePairs = [3][20][2]int{
	{ // COCO body
		{1, 2}, {1, 5}, {2, 3},
		{3, 4}, {5, 6}, {6, 7},
		{1, 8}, {8, 9}, {9, 10},
		{1, 11}, {11, 12}, {12, 13},
		{1, 0}, {0, 14},
		{14, 16}, {0, 15}, {15, 17},
	},
	{ // MPI body
		{0, 1}, {1, 2}, {2, 3},
		{3, 4}, {1, 5}, {5, 6},
		{6, 7}, {1, 14}, {14, 8}, {8, 9},
		{9, 10}, {14, 11}, {11, 12}, {12, 13},
	},
	{ // hand
		{0, 1}, {1, 2}, {2, 3}, {3, 4}, // thumb
		{0, 5}, {5, 6}, {6, 7}, {7, 8}, // pinkie
		{0, 9}, {9, 10}, {10, 11}, {11, 12}, // middle
		{0, 13}, {13, 14}, {14, 15}, {15, 16}, // ring
		{0, 17}, {17, 18}, {18, 19}, {19, 20}, // small
	}}
