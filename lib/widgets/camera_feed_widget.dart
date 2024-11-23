import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraFeedWidget extends StatefulWidget {
  final bool isCameraFeedOn;

  CameraFeedWidget({required this.isCameraFeedOn});

  @override
  _CameraFeedWidgetState createState() => _CameraFeedWidgetState();
}

class _CameraFeedWidgetState extends State<CameraFeedWidget> {
  CameraController? _controller;
  List<CameraDescription>? cameras;
  bool isCameraInitialized = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  // Initialize the camera
  Future<void> _initializeCamera() async {
    cameras = await availableCameras();
    if (cameras!.isNotEmpty) {
      _controller = CameraController(
        cameras![0], // Use the first camera
        ResolutionPreset.high,
      );
      await _controller?.initialize();
      setState(() {
        isCameraInitialized = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!isCameraInitialized) {
      return Container(
        height: 250,
        color: Colors.grey.shade300,
        child: const Center(child: CircularProgressIndicator()),
      );
    }

    return Container(
      height: 250,
      decoration: BoxDecoration(
        color: Colors.grey.shade300,
        borderRadius: BorderRadius.circular(16),
      ),
      child: widget.isCameraFeedOn
          ? CameraPreview(_controller!) // Display camera feed
          : const Center(
              child: Text(
                'Camera Off',
                style: TextStyle(fontSize: 24, color: Colors.black54),
              ),
            ),
    );
  }

  @override
  void dispose() {
    super.dispose();
    _controller?.dispose(); // Don't forget to dispose of the camera controller
  }
}
