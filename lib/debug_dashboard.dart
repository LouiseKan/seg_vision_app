import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;

class ModelUtils {
  static Future<Uint8List?> convertMaskToImage(List<List<num>> mask, int width, int height) async {
    final Uint8List pixels = Uint8List(width * height * 4);
    final colors = [
      [0, 0, 0, 0], [0, 255, 0, 150], [255, 0, 0, 150], [0, 0, 255, 150], [255, 255, 0, 150]
    ];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int label = mask[y][x].toInt();
        int index = (y * width + x) * 4;
        List<int> color = (label < colors.length) ? colors[label] : [255, 255, 255, 150];
        pixels[index] = color[0]; pixels[index + 1] = color[1]; pixels[index + 2] = color[2]; pixels[index + 3] = color[3];
      }
    }
    final Completer<ui.Image> completer = Completer();
    ui.decodeImageFromPixels(pixels, width, height, ui.PixelFormat.rgba8888, (img) => completer.complete(img));
    final ui.Image image = await completer.future;
    final ByteData? byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    return byteData?.buffer.asUint8List();
  }
}

class DebugDashboard extends StatefulWidget {
  final CameraController controller;
  final OrtSession session;
  const DebugDashboard({super.key, required this.controller, required this.session});
  @override
  State<DebugDashboard> createState() => _DebugDashboardState();
}

class _DebugDashboardState extends State<DebugDashboard> {
  Uint8List? _maskOverlay;
  Set<int> _detectedLabels = {};
  int _inferenceTime = 0;
  bool _isRunning = false;

  void _runDebugInference() async {
    if (_isRunning) return;
    setState(() => _isRunning = true);
    try {
      final stopwatch = Stopwatch()..start();
      XFile file = await widget.controller.takePicture();
      Uint8List bytes = await file.readAsBytes();
      img.Image? baseImage = img.decodeImage(bytes);
      if (baseImage == null) return;
      img.Image resized = img.copyResize(baseImage, width: 512, height: 512);

      var floatBuffer = Float32List(1 * 3 * 512 * 512);
      for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 512; x++) {
          var pixel = resized.getPixel(x, y);
          // 加入歸一化 (Normalization)
          floatBuffer[0 * 512 * 512 + y * 512 + x] = (pixel.r / 255.0 - 0.485) / 0.229;
          floatBuffer[1 * 512 * 512 + y * 512 + x] = (pixel.g / 255.0 - 0.456) / 0.224;
          floatBuffer[2 * 512 * 512 + y * 512 + x] = (pixel.b / 255.0 - 0.406) / 0.225;
        }
      }

      final inputName = widget.session.inputNames.first;
      final inputTensor = OrtValueTensor.createTensorWithDataList(floatBuffer, [1, 3, 512, 512]);
      final outputs = await widget.session.run(OrtRunOptions(), {inputName: inputTensor});

      if (outputs != null && outputs.isNotEmpty) {
        final dynamic data = outputs[0]?.value;
        if (data is List && data.isNotEmpty) {
          final mask = data[0] as List<List<num>>;
          Set<int> labels = {};
          for (var row in mask) { for (var p in row) { labels.add(p.toInt()); } }
          final maskImg = await ModelUtils.convertMaskToImage(mask, 512, 512);
          setState(() {
            _maskOverlay = maskImg;
            _detectedLabels = labels;
            _inferenceTime = stopwatch.elapsedMilliseconds;
          });
        }
      }
      inputTensor.release();
    } catch (e) { debugPrint("Debug Error: $e"); }
    finally { setState(() => _isRunning = false); }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Developer Debug Dashboard')),
      body: Column(
        children: [
          Expanded(
            flex: 3,
            child: Stack(
              children: [
                CameraPreview(widget.controller),
                if (_maskOverlay != null)
                  Positioned.fill(child: IgnorePointer(child: Image.memory(_maskOverlay!, fit: BoxFit.fill))),
                if (_isRunning) const Center(child: CircularProgressIndicator()),
              ],
            ),
          ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 40),
            color: Colors.black87,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('推論耗時: $_inferenceTime ms', style: const TextStyle(color: Colors.greenAccent)),
                Text('偵測到的 ID: ${_detectedLabels.isEmpty ? "無" : _detectedLabels.toList().toString()}',
                    style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 20),
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child: ElevatedButton.icon(
                    onPressed: _isRunning ? null : _runDebugInference,
                    icon: const Icon(Icons.analytics),
                    label: const Text('執行手動分析'),
                  ),
                ),
              ],
            ),
          )
        ],
      ),
    );
  }
}