import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'screens/sign_language_recognition_screen.dart';

void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    home: SignLanguageRecognitionScreen(),
  ));
}

class CameraPreviewPage extends StatefulWidget {
  @override
  _CameraPreviewPageState createState() => _CameraPreviewPageState();
}

class _CameraPreviewPageState extends State<CameraPreviewPage> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    // Request permissions and initialize the camera
    _initializeCamera();
  }

// Request Camera Permissions
Future<void> requestPermissions() async {
  if (await Permission.camera.request().isGranted) {
    // Proceed with initializing the camera if permission is granted
    _initializeCamera();
  } else {
    // Handle the case where permissions are not granted
    print("Camera permission not granted!");
  }
}

  // Initialize the camera
  void _initializeCamera() async {
    // Request camera permissions before accessing the camera
    await requestPermissions();
    
    // Get list of available cameras
    final cameras = await availableCameras();
    
    // Select the first camera (front or rear based on your needs)
    final firstCamera = cameras.first;

    // Initialize the camera controller
    _controller = CameraController(
      firstCamera, 
      ResolutionPreset.high,
    );

    // Initialize the controller and ensure everything is ready before showing the feed
    _initializeControllerFuture = _controller.initialize();
    setState(() {});
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Camera Feed')),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          // If the camera is still initializing, show a loading spinner
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else {
            return CameraPreview(_controller);  // Display live camera feed
          }
        },
      ),
    );
  }
}


// ################################################################################################

// import 'package:flutter/material.dart';
// import 'package:firebase_core/firebase_core.dart';
// import 'sign_recognition_page.dart';  // Import your page for sign recognition

// void main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//   await Firebase.initializeApp();  // Initialize Firebase
//   runApp(MyApp());
// }

// class MyApp extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: 'Sign Recognition',
//       theme: ThemeData(primarySwatch: Colors.blue),
//       home: SignRecognitionPage(),  // Set your page as the home screen
//     );
//   }
// }

