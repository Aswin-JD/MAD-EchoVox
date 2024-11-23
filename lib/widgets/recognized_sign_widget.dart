import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class RecognizedSignWidget extends StatelessWidget {
  final String recognizedSign;
  final double confidence;

  RecognizedSignWidget({required this.recognizedSign, required this.confidence});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          'Recognized Sign:',
          style: GoogleFonts.lato(
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 10),
        RichText(
          text: TextSpan(
            style: GoogleFonts.lato(fontSize: 18, color: Colors.black),
            children: [
              TextSpan(
                text: '$recognizedSign ',
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              const TextSpan(
                text: 'with confidence of ',
              ),
              TextSpan(
                text: '${confidence.toStringAsFixed(2)}',
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Colors.blueAccent,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
