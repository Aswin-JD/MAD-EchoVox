import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import '../widgets/camera_feed_widget.dart';
import '../widgets/recognized_sign_widget.dart';

class SignLanguageRecognitionScreen extends StatefulWidget {
  @override
  _SignLanguageRecognitionScreenState createState() =>
      _SignLanguageRecognitionScreenState();
}

class _SignLanguageRecognitionScreenState
    extends State<SignLanguageRecognitionScreen> {
  bool isCameraFeedOn = true;
  String recognizedSign = "Loading..."; // Placeholder for recognized sign
  double confidence = 0.0; // Placeholder for confidence value
  final String backendUrl = 'http://192.0.0.2:8000/latest-recognition/';
  // final String backendUrl = 'http://192.168.0.100:8000/recognize-sign/';


  @override
  void initState() {
  super.initState();
  print('InitState called'); // Debug print
  _fetchRecognitionData(); // Start fetching data on screen load
  }

  Future<void> _fetchRecognitionData() async {
  try {
    final response = await http.get(Uri.parse(backendUrl));
    print(response.statusCode);
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      setState(() {
        recognizedSign = data['sign'];
        print(recognizedSign);
        confidence = data['confidence'];
        print(confidence);
      });
      print('Response: $data'); // Debugging line
    } else {
      print('Error: ${response.statusCode} - ${response.body}');
    }
  } catch (e) {
    print('Exception: $e');
  }
}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'EchoVox',
          style: GoogleFonts.lato(
            color: Colors.black,
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
        backgroundColor: Colors.lightBlue.shade50,
        elevation: 0,
        leading: const Icon(
          Icons.menu,
          color: Colors.black,
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 20),
              Text(
                'Sign Language Recognition',
                style: GoogleFonts.lato(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    'Camera Feed',
                    style: GoogleFonts.lato(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Switch(
                    value: isCameraFeedOn,
                    onChanged: (value) {
                      setState(() {
                        isCameraFeedOn = value;
                      });
                    },
                  ),
                ],
              ),
              const SizedBox(height: 20),
              CameraFeedWidget(isCameraFeedOn: isCameraFeedOn),
              const SizedBox(height: 20),
              RecognizedSignWidget(
                recognizedSign: recognizedSign,
                confidence: confidence,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
